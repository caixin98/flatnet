python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi.py with $1 distdataparallel=True -p
# python  train_multi.py with $1 -p