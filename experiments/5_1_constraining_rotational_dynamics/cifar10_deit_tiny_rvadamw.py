import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""PYTHONPATH=$PYTHONPATH:$base_dir torchrun --nproc_per_node 1 train.py --output $output_dir \
            --data-dir $data_dir --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 224 224 --amp \
            --model deit_tiny_patch16_224 --model-kwargs linear_cfg=slinear \
            --color-jitter 0.3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --train-interpolation bicubic --mixup 0.8 --cutmix 1.0 --reprob 0.25 --drop-path 0.1  \
            -b ${b} --opt rvadamw --lr $lr --opt-eps 1e-8 --weight-decay $wd --sched cosine --epochs 600 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5  \
            --log-wandb --wandb-kwargs project=constraining-smd name="\"deit_tiny_patch16_224_rvadamw_b_${b}_iter_${iter}_lr${lr}_wd${wd}_seed${seed}\"" --seed $seed  \
            --dynamics-logger-cfg '../../shared/utils/base_extended_logger_cfg.yaml'
    """

seeds = [ 0, 1, 2 ]
wd = 0.05

# Configure runs
def main():
    # Same lr, wd range as Figure 2 in AdamW paper, 121 runs total
    base_dir = os.getcwd()
    lrs = [ 0.0005 ]

    idx = 0
    for lr in lrs:
        for seed in seeds:
            if do_run(idx):
                print(f"===== Running iteration={idx} {lr=:0.3e} {wd=:0.3e} =====")
                subprocess.run(
                    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                    cwd=Path(base_dir) / "submodules" / "timm",
                    env={k: str(v) for k, v in dict(
                        # The keys and values need to be strings
                        **os.environ,  # Inherit original variables (e.g. conda)
                        base_dir=base_dir,
                        lr=f"{lr:0.6}",
                        wd=f"{wd:0.6}",
                        iter=idx,
                        data_dir=Path.home() / "data" / "datasets",
                        output_dir=Path.home() / "data" / "runs",
                        seed=seed,
                        b=64,
                    ).items()}
                )
            idx += 1


if __name__ == "__main__":
    # Arguments to run a subset of iterations (e.g. for parallelization or infra failures)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--range", nargs=2, type=int, default=None,
        help="Run a subrange of the iterations from A to B (inclusive)",
    )
    parser.add_argument(
        "--iters", nargs='+', type=int, default=None,
        help="Run the provided subset of iterations",
    )
    args = parser.parse_args()

    assert not (args.range and args.iters)
    if args.range:
        do_run = lambda idx: args.range[0] <= idx <= args.range[1]
    elif args.iters:
        do_run = lambda idx: idx in args.iters
    else:
        do_run = lambda idx: True

    main()