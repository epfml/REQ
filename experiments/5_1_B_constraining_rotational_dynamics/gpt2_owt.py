import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""PYTHONPATH=$PYTHONPATH:$base_dir torchrun --standalone --nproc_per_node=1 train.py ${cfg_path} --learning_rate=$lr --seed=$seed --wandb_run_name="gpt2-124M-$name-seed$seed"
"""

# Configure runs
def main():
    base_dir = os.getcwd()
    cfgs = [
        (6e-4 * 8, "rv-zero", "config/train_gpt2_20tpp_rwsl_cos0.py"),
        (6e-4 * 16, "rv-best", "config/train_gpt2_20tpp_rwsl_cos0.py"),
        (6e-4 * 8, "base", "config/train_gpt2_20tpp_cos0.py"),
    ]
    seeds = [42, 43, 44]

    idx = 0
    for seed in seeds:
        for lr, name, cfg_path in cfgs:
            if do_run(idx):
                print(f"===== Running iteration={idx} {seed=} {name=} {cfg_path=} =====")
                subprocess.run(
                    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                    cwd=Path(base_dir) / "submodules" / "nanoGPT",
                    env={k: str(v) for k, v in dict(
                        # The keys and values need to be strings
                        **os.environ,  # Inherit original variables (e.g. conda)
                        base_dir=base_dir,
                        lr=f"{lr:0.3e}",
                        idx=idx,
                        seed=seed,
                        name=name,
                        cfg_path=cfg_path,
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
