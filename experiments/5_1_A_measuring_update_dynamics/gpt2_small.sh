base_dir=""

cd $base_dir/submodules/nanoGPT
PYTHONPATH=$PYTHONPATH:$base_dir torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py --warmup_iters=0 --lrs_cls=constant --max_iters=100000 --min_lr=0.0 --wandb_run_name="gpt2-dynamics-small-const-s42" --seed=42 # Unstable default seed
PYTHONPATH=$PYTHONPATH:$base_dir torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py --warmup_iters=0 --lrs_cls=constant --max_iters=100000 --linear_cfg=wslinear --min_lr=0.0 --wandb_run_name="gpt2-dynamics-ws-small-const"