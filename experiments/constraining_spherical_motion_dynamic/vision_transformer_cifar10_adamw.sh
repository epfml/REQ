BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/timm

i=1
TMUX_SESSION=""
EXPERIMENT=""
DATA_DIR=""
PPATH="/opt/conda/bin/python"
CONDA_ENV=""

DNN_MODEL="deit_tiny_patch16_224"

tmux new -d -s $TMUX_SESSION
for seed in 0 1 2
do
    for weight_decay in 0.05
    do
        for learning_rate in  0.0005
        do 
            ch_id=$(($i%1))
            wand_name="${DNN_MODEL}_c10_${EXPERIMENT}_vision_transformer_seed_${seed}_bs_128_wd_${weight_decay//./_}_lr_${learning_rate//./_}_run_${i}"
            tmux new-window -t $TMUX_SESSION -n $i
            tmux send-keys  -t $TMUX_SESSION:$i.0 "conda activate $CONDA_ENV" Enter\;

            echo "PYTHONPATH=$PPATH:$BASE_DIR torchrun --rdzv_backend c10d --rdzv_endpoint localhost:$i --nnodes 1 --nproc_per_node 1 train.py --output $BASE_DIR/output/" \
            " --data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 224 224 --amp" \
            " --model $DNN_MODEL" \
            " --color-jitter 0.3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --train-interpolation bicubic --mixup 0.8 --cutmix 1.0 --reprob 0.25 --drop-path 0.1 " \
            " -b 64 --opt adamw --lr $learning_rate --opt-eps 1e-8 --weight-decay $weight_decay --sched cosine --epochs 600 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 " \
            "--log-wandb --wandb-kwargs project=4-2-constraining-smd name=$wand_name --seed $seed" \

            tmux wait-for -L $ch_id\; send-keys -t $TMUX_SESSION:$i.0 \
            "PYTHONPATH=$PPATH:$BASE_DIR torchrun --rdzv_backend c10d --rdzv_endpoint localhost:$i --nnodes 1 --nproc_per_node 1 train.py --output $BASE_DIR/output/" \
            " --data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 224 224 --amp" \
            " --model $DNN_MODEL" \
            " --color-jitter 0.3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --train-interpolation bicubic --mixup 0.8 --cutmix 1.0 --reprob 0.25 --drop-path 0.1 " \
            " -b 64 --opt adamw --lr $learning_rate --opt-eps 1e-8 --weight-decay $weight_decay --sched cosine --epochs 600 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 " \
            "--log-wandb --wandb-kwargs project=4-2-constraining-smd name=$wand_name --seed $seed" Enter "tmux wait-for -U $ch_id" Enter\; 
            i=$(($i+1))
        done
    done
done