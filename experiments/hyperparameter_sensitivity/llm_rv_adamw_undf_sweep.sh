BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/llm_baselines

i=1
TMUX_SESSION=""
EXPERIMENT=""
DATA_DIR=""
PPATH="/opt/conda/bin/python"
CONDA_ENV=""

values1=(1 2 3 4 5 6 7 8)
is_zero_mean=("TRUE" "TRUE" "TRUE" "TRUE" "TRUE" "TRUE" "TRUE" "TRUE")
scale_invaraince_type=("channel" "channel" "channel" "channel" "channel" "channel" "channel" "channel")
update_norm_decay_factors=(0.9999999999 0.9999999 0.99999 0.9999 0.999 0.9 0.0 0.99)

learning_rate=0.004
weight_decay=0.5

tmux new -d -s $TMUX_SESSION
for bath_size in 55
do
    for seed in  0 1 2
    do
        for optimizer in "rvadamw"
        do
            for idx in "${!values1[@]}"; do
                zero_mean=${is_zero_mean[$idx]}
                scale_invariance=${scale_invaraince_type[$idx]}
                update_norm_decay_factor=${update_norm_decay_factors[$idx]}

                ch_id=$(($i%1))
                wand_name="${EXPERIMENT}_wd_${weight_decay//./_}_lr_${learning_rate//./_}_sd_${seed}_${scale_invariance}_0m_${zero_mean}_undf_${update_norm_decay_factor//./_}_run_${i}"
                tmux new-window -t $TMUX_SESSION -n $i
                tmux send-keys  -t $TMUX_SESSION:$i.0 "conda activate $CONDA_ENV" Enter\;

                echo "PYTHONPATH=$PPATH:$BASE_DIR torchrun --nproc_per_node 1 src/main.py " \
                " --wandb --wandb_project llm-parameters-sensitivity --wandb_run_prefix $wand_name " \
                " --n_embd 768 --n_head 12 --n_layer 12 --batch_size $bath_size --sequence_length 512 --acc_steps 3 " \
                "--dropout 0.2 --iterations 15000 --warmup_percent 0.02 --opt $optimizer --lr $learning_rate --opt-kwargs betas=\"(0.9,0.95)\" weight_decay=$weight_decay scale_invariance=$scale_invariance zero_mean=$zero_mean update_norm_decay_factor=$update_norm_decay_factor" \
                " --scheduler-kwargs div_factor=1e2 final_div_factor=1e4 --linear_cfg wslinear --seed $seed"

                tmux wait-for -L $ch_id\; send-keys -t $TMUX_SESSION:$i.0 \
                "PYTHONPATH=$PPATH:$BASE_DIR torchrun --nproc_per_node 1 src/main.py " \
                " --wandb --wandb_project llm-parameters-sensitivity --wandb_run_prefix $wand_name " \
                " --n_embd 768 --n_head 12 --n_layer 12 --batch_size $bath_size --sequence_length 512 --acc_steps 3 " \
                "--dropout 0.2 --iterations 15000 --warmup_percent 0.02 --opt $optimizer --lr $learning_rate --opt-kwargs betas=\"(0.9,0.95)\" weight_decay=$weight_decay scale_invariance=$scale_invariance zero_mean=$zero_mean update_norm_decay_factor=$update_norm_decay_factor" \
                " --scheduler-kwargs div_factor=1e2 final_div_factor=1e4 --linear_cfg wslinear --seed $seed" Enter "tmux wait-for -U $ch_id" Enter\;
                i=$(($i+1))
            done
        done
    done
done