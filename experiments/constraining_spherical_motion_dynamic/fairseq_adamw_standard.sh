BASE_DIR=$(pwd)
cd $BASE_DIR/submodules/fairseq

i=1
TMUX_SESSION=""
EXPERIMENT=""
DATA_DIR=""
PPATH="/opt/conda/bin/python"
CONDA_ENV=""
DATA_ROOT=""

FAIRSEQ_ROOT=$BASE_DIR/submodules/fairseq
IWSLT14=$DATA_ROOT/iwslt14

optimizer="adamw"
learning_rate=0.0005
weight_decay=0.0001
cf_linear="standard"

tmux new -d -s $TMUX_SESSION
for seed in 0 1 2
do
    ch_id=$(($i%1))
    tmux new-window -t $TMUX_SESSION -n $i
    tmux send-keys  -t $TMUX_SESSION:$i.0 "conda activate $CONDA_ENV" Enter\;

    tmux wait-for -L $ch_id\; send-keys -t $TMUX_SESSION:$i.0 \
    "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$BASE_DIR fairseq-train " \
    " $IWSLT14/bin/iwslt14.tokenized.de-en " \
    " --arch transformer_iwslt_de_en --share-decoder-input-output-embed" \
    " --optimizer $optimizer --adam-betas '(0.9, 0.98)' --clip-norm 0.0" \
    " --lr $learning_rate --lr-scheduler cosine --warmup-updates 4000" \
    " --dropout 0.3 --weight-decay $weight_decay" \
    " --criterion label_smoothed_cross_entropy --label-smoothing 0.1" \
    " --max-tokens 4096" \
    " --eval-bleu" \
    " --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}'" \
    " --eval-bleu-detok moses" \
    " --eval-bleu-remove-bpe" \
    " --eval-bleu-print-samples" \
    " --best-checkpoint-metric bleu --maximize-best-checkpoint-metric" \
    " --linear-cfg $cf_linear" \
    " --max-update 22021" \
    " --save-dir runs/${optimizer}_${cf_linear}_wd_${weight_decay//./_}_lr_${learning_rate//./_}_seed_${seed//./_}_wdz=False" \
    " --amp " \
    " --weight-decay-zero-bias " \
    " --wandb-project fairseq-4-2-section " \
    " --seed $seed " Enter "tmux wait-for -U $ch_id" Enter\; 
    i=$(($i+1))
done