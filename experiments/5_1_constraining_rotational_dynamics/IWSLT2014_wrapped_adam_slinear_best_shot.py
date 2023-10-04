import argparse
import os
from pathlib import Path
import subprocess

# Base shell command
shell_cmd = r"""CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$base_dir fairseq-train  \
    $iwslt14/bin/iwslt14.tokenized.de-en  \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer rvwrapper --wrapper-inner-type adam --wrapper-etar-func adamw --wrapper-hyper-parameters '{"betas" : [0.9, 0.98] }' \
    --clip-norm 0.0 \
    --lr $lr --lr-scheduler cosine --warmup-updates 4000 \
    --dropout 0.3 --weight-decay $wd \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --linear-cfg $cf_linear \
    --max-update 22021 \
    --save-dir runs/rvwrapper_${cf_linear}_wd_${wd//./_}_lr_${lr//./_}_seed_${seed//./_} \
    --amp \
    --wandb-project adam-wrapper-runs  \
    --seed $seed
 """

lr=0.0005
wds= [ 0.2 ]

# Configure runs
def main():
    base_dir = os.getcwd()
    seeds = [ 0, 1, 2 ]
    idx = 0
    for wd in wds:
        for seed in seeds:
            if do_run(idx):
                print(f"===== Running iteration={idx} {lr=:0.3e} {wd=:0.3e} =====")
                subprocess.run(
                    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                    cwd=Path(base_dir) / "submodules" / "fairseq",
                    env={k: str(v) for k, v in dict(
                        # The keys and values need to be strings
                        **os.environ,  # Inherit original variables (e.g. conda)
                        base_dir=base_dir,
                        lr=f"{lr:0.6}",
                        wd=f"{wd:0.6}",
                        iter=idx,
                        seed=seed,
                        output_dir=Path.home() / "data" / "runs" / "fairseq",
                        iwslt14=Path.home() / "data" / "iwslt14",
                        cf_linear='slinear',
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
