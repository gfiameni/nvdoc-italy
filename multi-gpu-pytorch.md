# Multi-node-training on slurm with PyTorch

### What's this?
* A simple note for how to start multi-node-training on slurm scheduler with PyTorch.
* Useful especially when scheduler is too busy that you cannot get multiple GPUs allocated, 
or you need more than 4 GPUs for a single job.
* Requirement: Have to use PyTorch [DistributedDataParallel(DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for this purpose.
* Warning: might need to re-factor your own code.
* Warning: might be secretly condemned by your colleagues because using too many GPUs. 

### Setup python script
* create a file `main.py` for example:
```python
import os
import builtins
import argparse
import torch
import numpy as np 
import random
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args
                                         
def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    ### model ###
    model = MyModel()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    ### optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    ### resume training if necessary ###
    if args.resume:
        pass
    
    ### data ###
    train_dataset = MyDataset(mode='train')
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_dataset = MyDataset(mode='val')
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    torch.backends.cudnn.benchmark = True
    
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
        
        # adjust lr if needed #
        
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        if args.rank == 0: # only val and save on master node
            validate(val_loader, model, criterion, epoch, args)
            # save checkpoint if needed #

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    pass
    # only one gpu is visible here, so you can send cpu data to gpu by 
    # input_data = input_data.cuda() as normal
    
def validate(val_loader, model, criterion, epoch, args):
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)
```
* this script is already executable on single node 
(e.g. slurm's interactive mode by `salloc`, e.g. with 2 GPUs) by 
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 main.py --net resnet18 \
--lr 1e-3 --epochs 50 --other_args
```
* alternatively it can be executed with slurm, see below

### Setup slurm script
* create a file `exp.sh` as follows:
```sh
#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --partition=gpu
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=p40&gmem24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --chdir=/scratch/shared/beegfs/your_dir/
#SBATCH --output=/scratch/shared/beegfs/your_dir/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

### the command to run
srun python main.py --net resnet18 \
--lr 1e-3 --epochs 50 --other_args

```
* run command in cluster `sbatch exp.sh`

### Reference & Acknowledgement
* pytorch classification example: https://github.com/pytorch/vision/blob/7b9d30eb7c4d92490d9ac038a140398e0a690db6/references/classification/train.py 
* facebook's MoCo example: https://github.com/facebookresearch/moco/blob/master/main_lincls.py
* Thank Bruno for some suggestions
* Last updated on Oct 2021
