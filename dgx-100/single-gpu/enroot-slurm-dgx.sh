#!/bin/bash

module load profile/deeplrn autoload pytorch
srun -N1 --cpus-per-task=8 --partition=m100_usr_prod --gres=gpu:volta:2 -A  --time=02:00:00 

python3 -m venv env
python -m pip install -U pip
python -m pip install -U pandas



scontrol show jobid -dd <jobid>

sstat --format=AveCPU,AvePages,AveRSS,AveVMSize,JobID -j <jobid> --allsteps

srun --ntasks=1 --nodes=1 --cpus-per-task=1 --partition=m100_all_serial -A cin_cadl --time=2:00:00 --gres=gpu:1 --pty /bin/bash

ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9



OPENBLAS=$OPENBLAS_LIB/libopenblas.so python -m pip install -U --no-cache-dir scipy


docker run -it --shm-size=256m

LD_DEBUG=all


git branch
git rev-parse --abbrev-ref HEAD
