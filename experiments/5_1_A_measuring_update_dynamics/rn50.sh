base_dir=""
output_dir=""
data_dir=""
lr=0.1
wd=1e-4
wandb_project=req_iclr
wandb_name=rn50_dynamics_const
seed=0

PYTHONPATH=$PYTHONPATH:$base_dir torchrun --nproc_per_node 1 submodules/timm/train.py --output $output_dir \
--data-dir $data_dir/imagenet --dataset ImageFolder --num-classes 1000 --pin-mem --input-size 3 224 224 --workers 24 \
--train-transforms 'RandomResizedCrop=(224,)' 'RandomHorizontalFlip=(0.5,)' PILToTensor='{}' \
--test-transforms 'Resize=(256,)' 'CenterCrop=(224,)' 'PILToTensor={}' \
--model resnetv2_50 --model-kwargs zero_init_last=False --amp --channels-last \
-b 256 --opt momentum --lr $lr --momentum 0.9 --weight-decay $wd --sched step --sched-on-update --epochs 50 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 0 --checkpoint-hist 5 \
--log-wandb --wandb-kwargs project=$wandb_project name=$wandb_name --seed $seed \
--dynamics-logger-cfg "shared/utils/rotational_logger_smooth_cfg.yaml"
