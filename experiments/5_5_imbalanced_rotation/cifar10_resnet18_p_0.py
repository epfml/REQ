import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""PYTHONPATH=$PYTHONPATH:$base_dir torchrun --nproc_per_node 1 train.py --output $output_dir \
    --data-dir $data_dir --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0 \
    --model cifar_resnet --model-kwargs --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]'  \
    --train-split \"train[:0.9]-0\" --val-split \"train[0.9:]-0\"  \
    -b ${b} --opt $optimizer --opt-kwargs control_group_percentage=$percentage --lr-base $lr --lr-base-size 128 --lr-base-scale linear --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
    --log-wandb --wandb-kwargs project=imbalanced-rotation name="\"rn18_rvc_adamw_b_${b}_iter_${iter}_lr${lr}_wd${wd}_seed${seed}_p0\"" --seed $seed \
    --dynamics-logger-cfg '../../shared/utils/base_logger_cfg.yaml' 
"""

best_wd = 0.01

# Configure runs
def main():
    base_dir = os.getcwd()
    lrs = [ 0.0083, 0.025, 0.05, 0.1, 0.3, 0.9, 4.5 ]
    seeds = [ 0, 1, 2 ]
    percentages = [ 0.0 ]
    idx = 0
    for seed in seeds:
        for percentage in percentages:
            for lr in lrs:
                if do_run(idx):
                    print(f"===== Running iteration={idx} {lr=:0.3e} {best_wd=:0.3e} =====")
                    subprocess.run(
                        ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                        cwd=Path(base_dir) / "submodules" / "timm",
                        env={k: str(v) for k, v in dict(
                            # The keys and values need to be strings
                            **os.environ,  # Inherit original variables (e.g. conda)
                            base_dir=base_dir,
                            lr=f"{lr:0.6}",
                            wd=f"{best_wd:0.6}",
                            iter=idx,
                            data_dir=Path.home()/ "data" / "datasets",
                            output_dir=Path.home()/ "data" / "runs",
                            seed=seed,
                            percentage=percentage,
                            b=128,
                            optimizer="rvcadamw",
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