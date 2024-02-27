import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""PYTHONPATH=$PYTHONPATH:$base_dir torchrun --nproc_per_node 1 train.py --output $output_dir \
--data-dir $data_dir --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 \
--crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0 \
--model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' 'pre_activation=False' \
${opt_cfg} \
-b $b --lr $lr --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5  --warmup-epochs 5 --checkpoint-hist 1 \
--log-wandb --wandb-kwargs project=req_iclr_rnln name="\"new_rn18_special_${name}_seed${seed}\"" --seed $seed \
--dynamics-logger-cfg '../../shared/utils/rotational_logger_cfg.yaml'
"""

# Configure runs
def main():
    base_dir = os.getcwd()
    seeds = [42, 43, 44, 45, 46]
    bs = 256
    lr = 1.25e-2
    wd = 8.0e-2
    opt_cfgs = [
        ("srvadaml2", r"--opt rvwrapper --opt-kwargs inner_type=adam etar_func=adamw"),
        ("rvadamw", r"--opt rvwrapper --opt-kwargs inner_type=adamw"),
        ("adamw", r"--opt adamw"),
    ]

    idx = 0
    for seed in seeds:
        for name, opt_cfg in opt_cfgs:
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
                        b=bs,
                        name=name,
                        opt_cfg=opt_cfg,
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