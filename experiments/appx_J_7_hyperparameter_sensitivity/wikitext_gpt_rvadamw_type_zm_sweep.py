import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""PYTHONPATH=$PPATH:$base_dir torchrun --nproc_per_node 1 src/main.py \
    --wandb --wandb_project llm-sensitivity-sweep --wandb_run_prefix llm_rvadamw_iter_${iter}_lr${lr}_wd${wd}_seed${seed}_zm${zero_mean}_${scale_invariance}  \
    --n_embd 768 --n_head 12 --n_layer 12 --batch_size 55 --sequence_length 512 --acc_steps 3 --dropout 0.2 \
    --iterations 15000 --warmup_percent 0.02 --opt rvadamw --lr $lr \
    --opt-kwargs betas=\(0.9,0.95\) weight_decay=$wd scale_invariance=$scale_invariance zero_mean=$zero_mean  \
    --linear_cfg slinear \
    --scheduler-kwargs div_factor=1e2 final_div_factor=1e4 \
    --dynamics-logger-cfg '../../shared/utils/rotational_logger_cfg.yaml' \
    --logger_output_dir $output_dir \
    --seed $seed
 """

is_zero_mean=[False, True, False]
scale_invariance_types=["tensor", "tensor", "channel"]


lr=0.004
wd=0.5

# Configure runs
def main():
    base_dir = os.getcwd()
    seeds = [ 0, 1, 2 ]
    idx = 0
    for seed in seeds:
        for zm, inv_type in zip(is_zero_mean, scale_invariance_types):
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
                        output_dir=Path.home()/ "data" / "runs" / "llms",
                        zero_mean=zm,
                        scale_invariance=inv_type
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
