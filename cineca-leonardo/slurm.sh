#!/bin/bash
#SBATCH --job-name=DDP
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH -A <account> # saldo -b

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

master_port=`ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | sort | uniq | shuf`
export MASTER_PORT=$master_port

export WORLD_SIZE=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
module load profile/global cineca/3.0.0

### the command to run
torchrun --nproc_per_node $SLURM_NTASKS --nnodes $SLURM_JOB_NUM_NODES ddp.py --num_epochs 5
