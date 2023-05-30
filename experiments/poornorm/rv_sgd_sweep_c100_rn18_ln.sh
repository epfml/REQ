# C100 RN18 LN with rvsgd, sweep lr
BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/timm

DATA_DIR=""

for seed in 23 24 20 21 22
do
    for lr in 0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.5 1.0 2.0 3.0
    do
        PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $BASE_DIR/output \
        --data-dir $DATA_DIR --dataset torch/CIFAR100 --dataset-download --num-classes 100 --pin-mem --input-size 3 32 32 --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --smoothing 0.0 --train-split "train[:0.9]-$seed" --val-split "train[0.9:]-$seed" \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' norm_cfg="{\"subtype\":\"gn\",\"dkwargs\":{\"G\":1}}" \
        -b 256 --opt rvsgd --lr $lr --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
        --log-wandb --wandb-kwargs project=melo_poor_norm name="\"c100-val-rn18-b256-ln-rvsgd-lr$lr-s$seed\"" 'tags=["c100_val_rn18_b256_ln_rvsgd_sweep_lr"]' --seed $seed
    done
done
