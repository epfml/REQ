import os
from pathlib import Path
import subprocess

shell_cmd = r"""PYTHONPATH=$PYTHONPATH:$base_dir torchrun --nproc_per_node 1 train.py --output $output_dir \
--data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train[:0.9]-$seed" --val-split "train[0.9:]-$seed" \
--train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
--test-transforms 'PILToTensor={}' \
--model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="{\"subtype\":\"gn\",\"dkwargs\":{\"G\":1}}" ${layer_args} \
-b 256 --opt momentum --lr $lr --weight-decay $wd --momentum 0.9 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
--log-wandb --wandb-kwargs project=req_iclr_rnln name="\"measure-neurons-c100-rn18-ln-${name}\"" 'tags=["c100_val_rn18_wsln_sgdm_sweep_lr2"]' --seed $seed \
--dynamics-logger-cfg "../../shared/utils/rotational_logger_heavy_disk_cfg.yaml"
"""

base_dir = os.getcwd()
seed = 42
lr = 0.2
wd = 5e-4

name = 'base'
layer_args = ""
subprocess.run(
    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
    cwd=Path(base_dir) / "submodules" / "timm",
    env={k: str(v) for k, v in dict(
        # The keys and values need to be strings
        **os.environ,  # Inherit original variables (e.g. conda)
        base_dir=base_dir,
        lr=f"{lr:0.3e}",
        wd=f"{wd:0.3e}",
        data_dir=Path.home()/"datasets",
        output_dir=Path.home()/"runs",
        seed=seed,
        name=name,
        layer_args=layer_args,
    ).items()}
)

name = 'ws'
layer_args = r'conv_cfg="{\"subtype\":\"wsconv2d\",\"dkwargs\":{\"gain\":False}}" linear_cfg="{\"subtype\":\"wslinear\",\"dkwargs\":{\"gain\":False}}"'
subprocess.run(
    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
    cwd=Path(base_dir) / "submodules" / "timm",
    env={k: str(v) for k, v in dict(
        # The keys and values need to be strings
        **os.environ,  # Inherit original variables (e.g. conda)
        base_dir=base_dir,
        lr=f"{lr:0.3e}",
        wd=f"{wd:0.3e}",
        data_dir=Path.home()/"datasets",
        output_dir=Path.home()/"runs",
        seed=seed,
        name=name,
        layer_args=layer_args,
    ).items()}
)
