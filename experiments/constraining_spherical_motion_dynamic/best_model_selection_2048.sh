BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/timm

i=1
TMUX_SESSION=""
EXPERIMENT=""
DATA_DIR=""
PPATH="/opt/conda/bin/python"
CONDA_ENV=""

DNN_MODEL="cifar_rn20"

tmux new -d -s $TMUX_SESSION

values1=(1 2 3 4 5 6)
weight_decays2=(0.01 0.0001 1.0 0.01 0.0001 1.0)
learning_rates3=(0.8 16 0.08 0.16 4.8 0.016)
optimizers=("rvadamw" "rvsgd" "rvlion" "adamw" "momentum" "lion")

for bath_size in 2048 
do
    for seed in 0 1 2
    do
        for idx in "${!values1[@]}"; do
            weight_decay=${weight_decays2[$idx]}
            learning_rate=${learning_rates3[$idx]}
            optimizer=${optimizers[$idx]}

            ch_id=$(($i%1))
            wand_name="${DNN_MODEL}_c10_${optimizer}_baseline_seed_${seed}_bs_${bath_size}_wd_${weight_decay//./_}_lr_${learning_rate//./_}_run_${i}"
            tmux new-window -t $TMUX_SESSION -n $i
            tmux send-keys  -t $TMUX_SESSION:$i.0 "conda activate $CONDA_ENV" Enter\;

            echo "PYTHONPATH=$PPATH:$BASE_DIR torchrun --rdzv_backend c10d --rdzv_endpoint localhost:$i --nnodes 1 --nproc_per_node 1 train.py --output $BASE_DIR/output/" \
            " --data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0" \
            " --model cifar_resnet --model-kwargs name=$DNN_MODEL " \
            " -b $bath_size --opt $optimizer --momentum 0.9 --lr $learning_rate --weight-decay $weight_decay --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1" \
            " --log-wandb --wandb-kwargs project=4-2-constraining-smd-cifar10 name='$wand_name' --seed $seed" \
           
            tmux wait-for -L $ch_id\; send-keys -t $TMUX_SESSION:$i.0 \
            "PYTHONPATH=$PPATH:$BASE_DIR torchrun --rdzv_backend c10d --rdzv_endpoint localhost:$i --nnodes 1 --nproc_per_node 1 train.py --output $BASE_DIR/output/" \
            " --data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0" \
            " --model cifar_resnet --model-kwargs name=$DNN_MODEL " \
            " -b $bath_size --opt $optimizer --momentum 0.9 --lr $learning_rate --weight-decay $weight_decay --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1" \
            " --log-wandb --wandb-kwargs project=4-2-constraining-smd-cifar10-full name='$wand_name' --seed $seed" Enter "tmux wait-for -U $ch_id" Enter\; 
            i=$(($i+1))
        done
    done
done