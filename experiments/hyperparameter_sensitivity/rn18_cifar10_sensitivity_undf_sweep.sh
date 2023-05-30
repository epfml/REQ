BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/timm

i=1
TMUX_SESSION=""
EXPERIMENT=""
DATA_DIR=""
PPATH="/opt/conda/bin/python"
CONDA_ENV=""

DNN_MODEL="resnet18"


values1=(1 2 3 4 5 6 7 8)
is_zero_mean=("TRUE" "TRUE" "TRUE" "TRUE" "TRUE" "TRUE" "TRUE" "TRUE")
scale_invariance_type=("channel" "channel" "channel" "channel" "channel" "channel" "channel" "channel")
update_norm_decay_factors=(0.9999999999 0.9999999 0.99999 0.9999 0.999 0.9 0.0 0.99)

weight_decay=0.0001
learning_rate=0.5

tmux new -d -s $TMUX_SESSION
for bath_size in 128
do
    for seed in  0 1 2
    do
        for optimizer in "rvsgd"
        do
            for idx in "${!values1[@]}"; do
                zero_mean=${is_zero_mean[$idx]}
                scale_invariance=${scale_invariance_type[$idx]}
                update_norm_decay_factor=${update_norm_decay_factors[$idx]}

                ch_id=$(($i%1))
                wand_name="cifar_${DNN_MODEL}_c10_${optimizer}_ablation_${scale_invariance}_0mean_${zero_mean}_update_norm_decay_factor_${update_norm_decay_factor//./_}_seed_${seed}_bs_${bath_size}_wd_${weight_decay//./_}_lr_${learning_rate//./_}_run_${i}_split_0_9_s0"
                tmux new-window -t $TMUX_SESSION -n $i
                tmux send-keys  -t $TMUX_SESSION:$i.0 "conda activate $CONDA_ENV" Enter\;

                echo "PYTHONPATH=$PPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $BASE_DIR/output/" \
                " --data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0" \
                " --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' " \
                " --train-split \"train[:0.9]-0\" --val-split \"train[0.9:]-0\" " \
                " -b $bath_size --opt $optimizer --opt-kwargs zero_mean=$zero_mean scale_invariance=$scale_invariance update_norm_decay_factor=$update_norm_decay_factor --momentum 0.9 --lr-base $learning_rate --lr-base-size 128 --lr-base-scale linear --weight-decay $weight_decay --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1" \
                " --log-wandb --wandb-kwargs project=5-6-sensitivity-sweep name='$wand_name' --seed $seed" \

                tmux wait-for -L $ch_id\; send-keys -t $TMUX_SESSION:$i.0 \
                "PYTHONPATH=$PPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $BASE_DIR/output/" \
                " --data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0" \
                " --model cifar_resnet --model-kwargs 'input_block=cifar' 'block_name=basicblock' 'stage_depths=[2,2,2,2]' " \
                " --train-split \"train[:0.9]-0\" --val-split \"train[0.9:]-0\" " \
                " -b $bath_size --opt $optimizer --opt-kwargs zero_mean=$zero_mean scale_invariance=$scale_invariance update_norm_decay_factor=$update_norm_decay_factor --momentum 0.9 --lr-base $learning_rate --lr-base-size 128 --lr-base-scale linear --weight-decay $weight_decay --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1" \
                " --log-wandb --wandb-kwargs project=5-6-sensitivity-sweep name='$wand_name' --seed $seed" Enter "tmux wait-for -U $ch_id" Enter\; 
                i=$(($i+1))
            done
        done
    done
done