# slurm-example.sh
#!/bin/bash
#SBATCH -A sa
#SBATCH -p luna
#SBATCH --time 00:15:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 8 tasks
#SBATCH --job-name=sa-ek100:test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email>
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j


export MY_SCRATCH=/lustre/fsw/sa
export TMPDIR=$MY_SCRATCH/$USER/tmp

#mkdir -p $TMPDIR

export ENROOT_CACHE_PATH=$MY_SCRATCH/$USER/enroot/tmp/enroot-cache
export ENROOT_DATA_PATH=$MY_SCRATCH/$USER/enroot/tmp/enroot-data
export ENROOT_RUNTIME_PATH=$MY_SCRATCH/$USER/enroot/tmp/enroot-runtime
export ENROOT_MOUNT_HOME=y NVIDIA_DRIVER_CAPABILITIES=all

export ENROOT_TEMP_PATH=/tmp

enroot start --mount $PWD:/workspace --mount /lustre/fsw/sa/ek-challenge/data:/data --root --env NVIDIA_DRIVER_CAPABILITIES --rw pytorch_2202 python -c 'import torch; print(torch.__version__); wait'

#!/bin/bash
#SBATCH -A IscrC_LSMAP-AI
#SBATCH -p dgx_usr_prod
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=8 # 8 tasks
#SBATCH --gres=gpu:1        # 1 gpus per node out of 8
#SBATCH --mem=7100          # memory per node out of 980000 MB
#SBATCH --job-name=mosaic
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email>
#SBATCH --nodelist=dgx3

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out

ssh login.dgx.cineca.it -l gfiamen1

srun -p dgx_usr_prod -n4 --gres=gpu:1 -t 02:00:00 -A cin_extern02_0 --nodelist dgx01 --pty /bin/bash

srun -p dgx_usr_prod -n4 --gres=gpu:1 -t 00:40:00 -A cin_extern02_0 --nodelist dgx02 --container-image=nvcr.io/nvidia/tensorflow:21.03-tf2-py3 --pty /bin/bash

export ENROOT_CACHE_PATH=$CINECA_SCRATCH/enroot-cache
export ENROOT_DATA_PATH=$CINECA_SCRATCH/enroot-data
export ENROOT_RUNTIME_PATH=$CINECA_SCRATCH/mosaic/enroot-run

export NVIDIA_DRIVER_CAPABILITIES=all
export ENROOT_CACHE_PATH=/raid/scratch_local/mosaic/tiramisu/enroot-cache
export ENROOT_DATA_PATH=/raid/scratch_local/mosaic/tiramisu/enroot-data
export ENROOT_RUNTIME_PATH=/raid/scratch_local/mosaic/tiramisu/enroot-runtime

srun -p dgx_usr_prod -n4 --gres=gpu:1 -t 02:00:00 -A IscrC_LSMAP-AI --nodelist dgx01 --pty /bin/bash

srun -p dgx_usr_prod -n2 --cpus-per-task=16 --gres=gpu:2 -t 01:00:00 -A cin_extern02_0 --nodelist dgx01 --pty /bin/bash

enroot import -o pytorch_2010.sqsh 'docker://@nvcr.io#nvidia/pytorch:20.10-py3'


enroot create pytorch_2010.sqsh

enroot start --mount $PWD:/tiramisu --root --env NVIDIA_DRIVER_CAPABILITIES --rw pytorch_2010


export ENROOT_CACHE_PATH=/raid/scratch_local/$USER/enroot/tmp/enroot-cache/group-$(id -g)
export ENROOT_DATA_PATH=/raid/scratch_local/$USER/enroot/tmp/enroot-data/user-$(id -u)
export ENROOT_RUNTIME_PATH=/raid/scratch_local/$USER/enroot/tmp/enroot-runtime/user-$(id -u)
export ENROOT_MOUNT_HOME=y NVIDIA_DRIVER_CAPABILITIES=all


enroot import -o pytorch_2008.sqsh 'docker://@nvcr.io#nvidia/pytorch:20.08-py3'


mpirun -n 1 --map-by node -mca pmix_base_async_modex 1 -mca mpi_add_procs_cutoff 0 -mca pmix_base_collect_data 0 --allow-run-as-root python trivial.py

horovodrun -n 1 python trivial.py

# To do once
# enroot import -o pytorch_2012.sqsh 'docker://@nvcr.io#nvidia/pytorch:20.12-py3'
# enroot create pytorch_2012.sqsh

DGX02
/raid/scratch_local/gfiamen1/prace/DeepLearningExamples


enroot start --mount $PWD:/unitn --root --env NVIDIA_DRIVER_CAPABILITIES --rw pytorch_2012 python -c 'import torch; print(torch.__version__)'

root@dgx02:/workspace/examples/resnet50v1.5# ./resnet50v1.5/

enroot import -o pytorch_2102.sqsh 'docker://@nvcr.io#nvidia/pytorch:21.02-py3'

enroot start --mount /raid/DATASETS_AI/imagenet:/imagenet --root --env NVIDIA_DRIVER_CAPABILITIES --rw pytorch_2102

enroot start --mount /raid/DATASETS_AI/imagenet:/imagenet --root --env NVIDIA_DRIVER_CAPABILITIES --rw tensorflow_2102

root@dgx02:/workspace/examples/resnet50v1.5# ./resnet50v1.5/training/TF32/DGXA100_RN50_TF32_90E.sh

export PYTHONWARNINGS="ignore"




export ENROOT_CACHE_PATH=$CINECA_SCRATCH/mosaic/enroot-cache
export ENROOT_DATA_PATH=$CINECA_SCRATCH/mosaic/enroot-data
export ENROOT_RUNTIME_PATH=$CINECA_SCRATCH/mosaic/enroot-run

https://www.glue.umd.edu/hpcc/help/software/pytorch.html


https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904


enroot start --mount $PWD:/workspace --root --env NVIDIA_DRIVER_CAPABILITIES --rw pytorch_2103 sh -c 'cd ek55 ; . ./scripts/train_anticipation_ek55.sh > output 2>&1 &'
