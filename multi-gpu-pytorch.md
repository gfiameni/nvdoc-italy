# Multi-node-training on slurm with PyTorch

### What's this?
* A simple note for how to start multi-node-training on slurm scheduler with PyTorch.
* Useful especially when scheduler is too busy that you cannot get multiple GPUs allocated, 
or you need more than 4 GPUs for a single job.
* Requirement: Have to use PyTorch [DistributedDataParallel(DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for this purpose.
* Warning: might need to re-factor your own code.
* Warning: might be secretly condemned by your colleagues because using too many GPUs. 

```
* this script is already executable on single node 
(e.g. slurm's interactive mode by `salloc`, e.g. with 2 GPUs) by 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 main.py --net resnet18 \
--lr 1e-3 --epochs 50 --other_args
* alternatively it can be executed with slurm, see below
```


mpirun ./myexecutable       #in case you compiled with spectrum-mpi
OR
srun ./myexecutable         #in all the other cases


### Setup slurm script
* create a file `exp.sh` as follows:
```sh
#!/bin/bash
#SBATCH -A cin_extern02
#SBATCH -p m100_usr_prod
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 8 tasks out of 128
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --mem=7100          # memory per node out of 246000MB
#SBATCH --job-name=my_batch_job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<user_email>

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --constraint=p40&gmem24G
#SBATCH --cpus-per-task=8
##SBATCH --mem=64gb
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

