DATA_SET_DIR=""
BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/timm

seed=0
PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $BASE_DIR/output \
--data-dir $DATA_SET_DIR --dataset ImageFolder --num-classes 1000 --pin-mem --input-size 3 224 224 --workers 24 \
--train-transforms 'RandomResizedCrop=(224,)' 'RandomHorizontalFlip=(0.5,)' PILToTensor='{}' \
--test-transforms 'Resize=(256,)' 'CenterCrop=(224,)' 'PILToTensor={}' \
--model resnet50 --model-kwargs zero_init_last=False --amp --channels-last \
-b 256 --opt momentum --lr 0.1 --momentum 0.9 --weight-decay 1e-4 --sched cosine --sched-on-update --epochs 90 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 1 --checkpoint-hist 5 \
--log-wandb --wandb-kwargs project=constraining-smd name=rn50_i1k_sgd_zero_init_${seed}  --seed $seed