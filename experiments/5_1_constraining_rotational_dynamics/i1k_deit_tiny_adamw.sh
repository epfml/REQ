DATA_SET_DIR=""
BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/timm

seed=0
PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --model deit_tiny_patch16_224 --output $BASE_DIR/output --amp \
--data-dir  $DATA_SET_DIR --dataset ImageFolder --num-classes 1000 --pin-mem --input-size 3 224 224 --workers 24 \
--color-jitter 0.3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --train-interpolation bicubic --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
--drop-path 0.1 -b 1024 --opt adamw --lr 5e-4 --lr-base-size 512 --opt-eps 1e-8 --weight-decay 0.05 --sched cosine --sched-on-update --epochs 300 \
--warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 \
--log-wandb --wandb-kwargs project=constraining-smd name=deit_tiny_i1k_adamw_base_${seed} --seed $seed