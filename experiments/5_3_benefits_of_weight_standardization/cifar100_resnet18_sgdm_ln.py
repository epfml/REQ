import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""PYTHONPATH=$PYTHONPATH:$base_dir torchrun --nproc_per_node 1 train.py --output $output_dir \
--data-dir $data_dir --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train[:0.9]-$seed" --val-split "train[0.9:]-$seed" \
--train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
--test-transforms 'PILToTensor={}' \
--model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="{\"subtype\":\"gn\",\"dkwargs\":{\"G\":1}}" \
-b 256 --opt momentum --lr $lr --weight-decay $wd --momentum 0.9 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
--log-wandb --wandb-kwargs project=req_iclr_rnln name="\"c100-val-rn18-ln-sgdm-wd$wd-s$seed-idx$idx\"" 'tags=["c100_val_rn18_ln_sgdm_sweep_lr"]' --seed $seed \
--dynamics-logger-cfg "../../shared/utils/rotational_logger_cfg.yaml"
"""

# Configure runs
def main():
    base_dir = os.getcwd()
    seeds = [42, 43, 44]
    lrs = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 0.01, 0.02, 0.03, 0.05, ]
    wd = 5e-4

    idx = 0
    for seed in seeds:
        for lr in lrs:
            if do_run(idx):
                print(f"===== Running iteration={idx} {lr=:0.3e} {wd=:0.3e} =====")
                subprocess.run(
                    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                    cwd=Path(base_dir) / "submodules" / "timm",
                    env={k: str(v) for k, v in dict(
                        # The keys and values need to be strings
                        **os.environ,  # Inherit original variables (e.g. conda)
                        base_dir=base_dir,
                        lr=f"{lr:0.3e}",
                        wd=f"{wd:0.3e}",
                        idx=idx,
                        data_dir=Path.home()/"datasets",
                        output_dir=Path.home()/"runs",
                        seed=seed,
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
