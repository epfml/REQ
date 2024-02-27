BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/timm

DATA_SET_DIR=""

for seed in 10
do
    for lr in 6.4
    do
        PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $BASE_DIR/output \
        --data-dir $DATA_SET_DIR --dataset ImageFolder --num-classes 1000 --pin-mem --input-size 3 224 224 --workers 24 \
        --train-transforms 'RandomResizedCrop=(224,)' 'RandomHorizontalFlip=(0.5,)' PILToTensor='{}' \
        --test-transforms 'Resize=(256,)' 'CenterCrop=(224,)' 'PILToTensor={}' \
        --model resnet50 --model-kwargs zero_init_last=False --amp --channels-last \
        -b 256 --local-accumulation 32 --opt rvwrapper --opt-kwargs "inner_type=sgdm" "etar_func=sgdm" --lr $lr --momentum 0.9 --weight-decay 1e-4 --sched cosine --sched-on-update --epochs 90 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 1 --checkpoint-hist 5 \
        --log-wandb --wandb-kwargs project=lr_warmup name="\"full-with-warm-up-i1k-rn50-nzi-b256x32-bn-rvsgd-lr$lr-s$seed\"" --dynamics-logger-cfg '../../shared/utils/sgd_tracked_logger_cfg_8.yaml' --seed $seed
    done
done
