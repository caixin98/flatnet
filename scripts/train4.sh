python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py with $1 distdataparallel=True -p
# python  train.py with $1 -p