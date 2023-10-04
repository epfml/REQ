import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(norm_str, atomic=False):
    assert not atomic
    if norm_str is None or norm_str.lower() == 'none':
        return nn.Identity
    if norm_str.lower() in ['batch', 'batch_norm', 'bn']:
        return nn.BatchNorm2d
    if re.search('group|gn|group_norm', norm_str.lower()):
        groups = re.search(r'\d+$', norm_str.lower())
        if groups is not None:
            groups = int(groups.group(0))
        else:
            groups = 8
        return lambda num_channels: nn.GroupNorm(groups, num_channels)

    raise ValueError(f"Unknown norm_str: {norm_str}")


class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 device=None, dtype=None, gain=True, keep_init=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("gain", None)

        # Keep the initialization magnitude, otherwise use fan-in
        self.keep_init = keep_init
        self.buffer_initialized = False
        if self.keep_init:
            self.register_buffer('init_std', torch.zeros(out_channels, device=device, dtype=dtype))

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=[1, 2, 3], keepdim=True) + 1e-5

        if self.keep_init and not self.buffer_initialized:
            with torch.no_grad():
                self.init_std.copy_(std.flatten())
            self.buffer_initialized = True
        
        if not self.keep_init:
            fan_in = weight.size(1) * weight.size(2) * weight.size(3)
            scale_factor = 1.0 / (std * math.sqrt(fan_in))
        else:
            scale_factor = self.init_std.view(-1, 1, 1, 1) / std
            
        if self.gain is not None:
            scale_factor = scale_factor * self.gain.view(-1, 1, 1, 1)
        weight = scale_factor * weight  # Could also apply to outputs, note different memory impact
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class WSLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, gain=True, keep_init=True):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if gain:
            self.gain = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("gain", None)

        # Keep the initialization magnitude, otherwise use fan-in
        self.keep_init = keep_init
        self.buffer_initialized = False
        if self.keep_init:
            self.register_buffer('init_std', torch.zeros(out_features, device=device, dtype=dtype))

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1], keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=[1], keepdim=True) + 1e-5

        if self.keep_init and not self.buffer_initialized:
            with torch.no_grad():
                self.init_std.copy_(std.flatten())
            self.buffer_initialized = True
        
        if not self.keep_init:
            fan_in = weight.shape[1]
            scale_factor = 1.0 / (std * math.sqrt(fan_in))
        else:
            scale_factor = self.init_std.view(-1, 1) / std

        if self.gain is not None:
            scale_factor = scale_factor * self.gain.view(-1, 1)
        weight = scale_factor * weight  # Could also apply to outputs, note different memory impact
        return torch.nn.functional.linear(x, weight, self.bias)
    
class SLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.gain = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype))

    def forward(self, x):
        weight = self.weight
        weight = self.gain.view(-1, 1) * weight  # Could also apply to outputs, note different memory impact
        return torch.nn.functional.linear(x, weight, self.bias)


class TrackingNorm(nn.Module):
    def __init__(self, num_features, rate=0.01, affine=True, channelwise=True):
        super().__init__()
        self.num_features = num_features
        self.rate = rate
        self.affine = affine
        self.channelwise = channelwise

        self.idx = 0

        if affine:
            self.bias = nn.Parameter(torch.zeros(num_features))
            self.gain = nn.Parameter(torch.ones(num_features))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("gain", None)

        tracker_size = num_features if channelwise else 1
        self.register_buffer('var_tracker', torch.ones(tracker_size))
        self.register_buffer('mean_tracker', torch.zeros(tracker_size))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                dims = [i for i in range(x.dim()) if i != 1] if self.channelwise else None
                mean = x.mean(dim=dims)
                var = x.var(dim=dims)

                self.mean_tracker.copy_(self.mean_tracker * (1-self.rate) + self.rate * mean)
                self.var_tracker.copy_(self.var_tracker * (1-self.rate) + self.rate * var)

            self.idx += 1
            if self.idx % 100 == 0:
                print(f"{self.mean_tracker=}, {self.var_tracker=}, {x.mean()=}, {x.var()=}")

        shape = [1]*x.dim()
        shape[1] = -1
        x_norm = (x - self.mean_tracker.view(*shape))/torch.sqrt(self.var_tracker.view(*shape) + 1e-5)
        if self.affine:
            x_norm = x_norm * self.gain.view(*shape) + self.bias.view(*shape)
        return x_norm

    def extra_repr(self):
        s = "{num_features}"
        s += ", rate={rate}"
        s += ", channelwise={channelwise}"
        s += ", affine={affine}"
        return s.format(**self.__dict__)


class TWSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 device=None, dtype=None, gain=True, keep_init=True, rate=0.01, channelwise=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.rate = rate
        if gain:
            self.gain = nn.Parameter(torch.ones(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("gain", None)

        self.channelwise = channelwise
        buffer_size = out_channels if channelwise else 1
        self.register_buffer('var_tracker', torch.ones(buffer_size))
        self.register_buffer('mean_tracker', torch.zeros(buffer_size))

        # Keep the initialization magnitude, otherwise use fan-in
        self.keep_init = keep_init
        self.buffer_initialized = False
        if self.keep_init:
            self.register_buffer('init_var', torch.zeros(buffer_size, device=device, dtype=dtype))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                dims = [1, 2, 3] if self.channelwise else None
                mean = self.weight.mean(dim=dims)
                var = self.weight.var(dim=dims)

            if not self.buffer_initialized:
                self.buffer_initialized = True
                self.mean_tracker.copy_(mean)
                self.var_tracker.copy_(var)
                if self.keep_init:
                    self.init_var.copy_(var)

        # Compute forward weight
        weight = self.weight - self.mean_tracker.view(-1, 1, 1, 1)
        if self.keep_init:
            weight = weight * torch.sqrt((self.init_var + 1e-6) / (self.var_tracker + 1e-6)).view(-1, 1, 1, 1)
        else:
            weight = weight / torch.sqrt(self.var_tracker + 1e-6).view(-1, 1, 1, 1)

        if self.gain is not None:
            weight = weight * self.gain.view(-1, 1, 1, 1)

        # Update trackers
        if self.training:
            self.mean_tracker.copy_(self.mean_tracker * (1-self.rate) + self.rate * mean)
            self.var_tracker.copy_(self.var_tracker * (1-self.rate) + self.rate * var)

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = super().extra_repr()
        s += f", rate={self.rate}"
        s += f", channelwise={self.channelwise}"
        s += f", keep_init={self.keep_init}"
        s += f", gain={self.gain is not None}"
        return s


class TWSLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, gain=True, keep_init=True, rate=0.01, channelwise=True):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.rate = rate
        if gain:
            self.gain = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("gain", None)

        self.channelwise = channelwise
        buffer_size = out_features if channelwise else 1
        self.register_buffer('var_tracker', torch.ones(buffer_size))
        self.register_buffer('mean_tracker', torch.zeros(buffer_size))

        # Keep the initialization magnitude, otherwise use fan-in
        self.keep_init = keep_init
        self.buffer_initialized = False
        if self.keep_init:
            self.register_buffer('init_var', torch.zeros(buffer_size, device=device, dtype=dtype))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                dims = [1] if self.channelwise else None
                mean = self.weight.mean(dim=dims)
                var = self.weight.var(dim=dims)

            if not self.buffer_initialized:
                self.buffer_initialized = True
                self.mean_tracker.copy_(mean)
                self.var_tracker.copy_(var)
                if self.keep_init:
                    self.init_var.copy_(var)

        # Compute forward weight
        weight = self.weight - self.mean_tracker.view(-1, 1)
        if self.keep_init:
            weight = weight * torch.sqrt((self.init_var + 1e-6) / (self.var_tracker + 1e-6)).view(-1, 1)
        else:
            weight = weight / torch.sqrt(self.var_tracker + 1e-6).view(-1, 1)

        if self.gain is not None:
            weight = weight * self.gain.view(-1, 1)

        # Update trackers
        if self.training:
            self.mean_tracker.copy_(self.mean_tracker * (1-self.rate) + self.rate * mean)
            self.var_tracker.copy_(self.var_tracker * (1-self.rate) + self.rate * var)

        return torch.nn.functional.linear(x, weight, self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        s += f", rate={self.rate}"
        s += f", channelwise={self.channelwise}"
        s += f", keep_init={self.keep_init}"
        s += f", gain={self.gain is not None}"
        return s
