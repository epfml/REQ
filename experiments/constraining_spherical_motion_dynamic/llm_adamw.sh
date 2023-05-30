BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/llm_baselines

i=1
TMUX_SESSION=""
EXPERIMENT=""
DATA_DIR=""
PPATH="/opt/conda/bin/python"
CONDA_ENV=""


tmux new -d -s $TMUX_SESSION
for lin_type in 'standard'
do
    for seed in 0 1 2
    do
        for weight_decay in 0.5
        do
            for learning_rate in 0.005
            do 
                ch_id=$(($i%1))
                wand_name="llm_${lin_type}_c10_${EXPERIMENT}_scheduled_bs_55_wd_${weight_decay//./_}_lr_${learning_rate//./_}_seed_${seed}_run_${i}"
                tmux new-window -t $TMUX_SESSION -n $i
                tmux send-keys  -t $TMUX_SESSION:$i.0 "conda activate $CONDA_ENV" Enter\;

                echo "PYTHONPATH=$PPATH:$BASE_DIR torchrun --nproc_per_node 1 src/main.py " \
                " --wandb --wandb_project llm-sweep-4-2-section --wandb_run_prefix $wand_name " \
                " --n_embd 768 --n_head 12 --n_layer 12 --batch_size 55 --sequence_length 512 --acc_steps 3 --dropout 0.2 " \
                " --iterations 15000 --warmup_percent 0.02 --opt adamw --lr $learning_rate --is-test-split --opt-kwargs \"betas=(0.9,0.95) weight_decay=$weight_decay\" " \
                " --linear_cfg $lin_type --scheduler-kwargs div_factor=1e2 final_div_factor=1e4 "

                tmux wait-for -L $ch_id\; send-keys -t $TMUX_SESSION:$i.0 \
                "PYTHONPATH=$PPATH:$BASE_DIR torchrun --nproc_per_node 1 src/main.py " \
                " --wandb --wandb_project llm-sweep-4-2-section --wandb_run_prefix $wand_name  " \
                " --n_embd 768 --n_head 12 --n_layer 12 --batch_size 55 --sequence_length 512 --acc_steps 3 --dropout 0.2 " \
                " --iterations 15000 --warmup_percent 0.02 --opt adamw --lr $learning_rate --is-test-split " \
                " --opt-kwargs \"betas=(0.9,0.95)\" \"weight_decay=$weight_decay\" --linear_cfg $lin_type --seed $seed" \
                " --scheduler-kwargs div_factor=1e2 final_div_factor=1e4" Enter "tmux wait-for -U $ch_id" Enter\; 
                i=$(($i+1))
            done
        done
    done
done