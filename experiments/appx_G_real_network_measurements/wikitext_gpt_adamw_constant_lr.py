import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""PYTHONPATH=$PPATH:$base_dir torchrun --nproc_per_node 1 src/main.py \
    --wandb --wandb_project real-network-tracking --wandb_run_prefix llm_standard_adamw_iter_${iter}_lr${lr}_wd${wd}_seed${seed}  \
    --n_embd 768 --n_head 12 --n_layer 12 --batch_size 55 --sequence_length 512 --acc_steps 3 --dropout 0.2 \
    --iterations 15000 --warmup_percent 0.02 --opt adamw --lr $lr \
    --opt-kwargs betas=\(0.9,0.95\) weight_decay=$wd --linear_cfg standard --seed $seed \
    --scheduler-kwargs div_factor=1.0 final_div_factor=1.0 \
    --dynamics-logger-cfg '../../shared/utils/base_extended_logger_cfg.yaml' \
    --logger_output_dir $output_dir
 """

wd = 0.5
lr = 0.0005

# Configure runs
def main():
    base_dir = os.getcwd()
    seeds = [ 0 ]

    idx = 0
    for seed in seeds:
        if do_run(idx):
            print(f"===== Running iteration={idx} {lr=:0.3e} {wd=:0.3e} =====")
            subprocess.run(
                ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                cwd=Path(base_dir) / "submodules" / "llm_baselines",
                env={k: str(v) for k, v in dict(
                    # The keys and values need to be strings
                    **os.environ,  # Inherit original variables (e.g. conda)
                    base_dir=base_dir,
                    lr=f"{lr:0.6}",
                    wd=f"{wd:0.6}",
                    iter=idx,
                    seed=seed,
                    output_dir=Path.home()/ "data" / "runs",
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