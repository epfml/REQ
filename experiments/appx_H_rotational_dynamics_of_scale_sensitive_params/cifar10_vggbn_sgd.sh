BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/timm

DATA_SET_DIR=""

lrs=(
  5e-1
  5e-2
  5e-3
  5e-4
)
wds=(
  5e-5
  5e-4
  5e-3
  5e-2
)

for index in ${!lrs[*]}; do
    lr=${lrs[$index]}
    wd=${wds[$index]}
    echo "$lr $wd"

    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $BASE_DIR/output \
        --data-dir $DATA_SET_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model vgg13_bn --model-kwargs num_features=512 feature_window_size=1 \
        -b 128 --opt momentum --lr $lr --momentum 0.9 --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 --seed 0 \
        --log-wandb --wandb-kwargs project=scale-sensitive-params name="\"sweep-vgg13bn-lr$lr-wd$wd-0\"" 

    PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $BASE_DIR/output \
        --data-dir $DATA_SET_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --smoothing 0.0 \
        --train-transforms 'RandomCrop=(32,4,)' 'RandomHorizontalFlip=(0.5,)' 'PILToTensor={}' \
        --test-transforms 'PILToTensor={}' \
        --model vgg13 --model-kwargs num_features=512 feature_window_size=1 conv_cfg="{\"subtype\":twsconv2d, \"dkwargs\":{\"channelwise\":False}}" linear_cfg="{\"subtype\":twslinear, \"dkwargs\":{\"channelwise\":False}}" \
        -b 128 --opt momentum --lr $lr --momentum 0.9 --weight-decay $wd --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 --seed 0 \
        --log-wandb --wandb-kwargs project=scale-sensitive-params name="\"sweep-vgg13tws-lr$lr-wd$wd-0\"" 
done
