import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""PYTHONPATH=$PYTHONPATH:$base_dir torchrun --standalone --nproc_per_node=1 train.py config/train_gpt2_20tpp_rwsl_cos0.py --learning_rate=$lr --warmup_iters=${warmup_iters} --wandb_run_name="gpt2-124M-rwsl-cos$wp-lr-scale$scale"
"""

# Configure runs
def main():
    base_dir = os.getcwd()
    warmups = [True, False]
    scales = [1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    lr = 6e-4

    idx = 0
    for scale in scales:
        for warmup in warmups:
            if do_run(idx):
                print(f"===== Running iteration={idx} {scale=:0.2e} {warmup=} =====")
                subprocess.run(
                    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                    cwd=Path(base_dir) / "submodules" / "nanoGPT",
                    env={k: str(v) for k, v in dict(
                        # The keys and values need to be strings
                        **os.environ,  # Inherit original variables (e.g. conda)
                        base_dir=base_dir,
                        scale=scale,
                        lr=f"{scale*lr:0.3e}",
                        wp=(5 if warmup else 0),
                        warmup_iters=(250 if warmup else 0),
                        idx=idx,
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
