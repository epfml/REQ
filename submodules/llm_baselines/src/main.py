import datetime
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import yaml
import wandb
from pathlib import Path

import config
from models.utils import get_model
from data.utils import get_dataset
from optim.base import train_base
from optim.sparse import train_sparse
import distributed

from shared.optimizers import optimizer_factories
from shared.utils.utils import call_valid_kwargs
from shared.utils.logger import DynamicsLogger

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def get_exp_name(args):
    """ Returns the name of the experiment, used for saving models and wandb. """
    exp_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.wandb_run_prefix != 'none':
        exp_name = args.wandb_run_prefix + '_' + exp_name
    return exp_name


def main(args): 
    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    exp_name = get_exp_name(args)
    ckpt_dir = Path(args.results_base_folder) / args.dataset / args.model / exp_name
    args.resumed = ckpt_dir.exists() and args.allow_resume
    if not ckpt_dir.exists():
        if distributed_backend.is_master_process():
            ckpt_dir.mkdir(parents=True)
    elif (ckpt_dir / "summary.json").exists():
        # the experiment was already completed
        print(f"Found completed experiment '{ckpt_dir}'.\nSkipping.")
        sys.exit(0)

    if distributed_backend.is_master_process() and args.wandb:
        if args.resumed:
            with open(ckpt_dir / 'wandb_id.txt', 'r') as f:
                wandbid = f.read()
            wandb.init(id=wandbid, project=args.wandb_project, resume='must')
        else:
            wandbid = wandb.util.generate_id()
            with open(ckpt_dir / 'wandb_id.txt', 'w') as f:
                f.write(wandbid)
            params_copy = copy.deepcopy(vars(args))
            wandb.init(id=wandbid, project=args.wandb_project, name=exp_name, config=params_copy)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset '{args.dataset}'")
    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
    if args.data_in_ram:
        data = {'train': np.array(data['train']), 'val': np.array(data['val'])}
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")

    model = get_model(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'
    model = distributed_backend.transform_model(model)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt/1e6,))
    print("model structre:", model)

    if args.opt.lower() in optimizer_factories:
        opt_kwargs = args.opt_kwargs
        opt_kwargs['lr'] = args.lr
        opt = call_valid_kwargs(optimizer_factories[args.opt.lower()], opt_kwargs, (group_specs,))
    elif args.opt == 'adamw':
        opt_kwargs = dict(
            # Default arguments
            betas=(0.9,0.95),
            weight_decay=1e-3,
        )
        opt_kwargs.update(args.opt_kwargs)
        opt_kwargs['lr'] = args.lr
        opt = torch.optim.AdamW(group_specs, **opt_kwargs)
    else:
        opt_kwargs = dict(
            # Default arguments
            weight_decay=1e-3,
            momentum=0.9,
        )
        opt_kwargs.update(args.opt_kwargs)
        opt_kwargs['lr'] = args.lr
        opt = torch.optim.SGD(group_specs, **args.opt_kwargs)

    # DynamicsLogger
    if args.dynamics_logger_cfg:
        with open(args.dynamics_logger_cfg, 'r') as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(model, opt, dlcfg, args.logger_output_dir)

    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            scheduler_kwargs = dict(
                optimizer=opt, max_lr=args.lr, total_steps=args.iterations, 
                pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                cycle_momentum=False, div_factor=1e2, final_div_factor=.05
            )
            scheduler_kwargs.update(args.scheduler_kwargs)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(**scheduler_kwargs)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    if args.model == 'base': # all train functions have the same interface
        train = train_base
    elif 'sparse' in args.model:
        train = train_sparse
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    print(f"model:\n{model}")
    print(f"optimizer:\n{opt}")
    print(f"scheduler:\n{scheduler}")
    print(f"Training model={args.model}\n{vars(args)}")
    print(f"Output dir: {ckpt_dir.resolve()}")

    stats = train(model, opt, data, scheduler, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  eval_freq=args.eval_freq, 
                  print_freq=args.print_freq,
                  distributed_backend=distributed_backend,
                  ckpt_dir=ckpt_dir, extra_args=args)

    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(ckpt_dir/"summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
