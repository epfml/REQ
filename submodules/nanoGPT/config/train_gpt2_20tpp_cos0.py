# PYTHONPATH=$PYTHONPATH:../../ torchrun --standalone --nproc_per_node=1 train.py config/train_gpt2_20tpp.py --learning_rate=6e-4 --wandb_run_name="gpt2-124M-trap5-10-lr-scale1"
# Scales: 0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0

wandb_log = True
wandb_project = 'owt3'
wandb_run_name='gpt2-124M-trap5-10-lr-6e-4'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 20B, ~20 tokens per parameter
max_iters = 5000
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
warmup_iters = 250 # how many steps to warm up for (5%)
min_lr = 0.0

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
