#!/bin/bash

#PBS -N diffseis_64x256_100k
#PBS -q gpu_TRIAL
#PBS -b 1
#PBS -r n
#PBS -l elapstim_req=2:00:00
#PBS -o stdout.%s.%j
#PBS -e stderr.%s.%j
#PBS --custom gpusetnum-lhost=1

cd $PBS_O_WORKDIR
#module load python torch
module load CUDA cuDNN
module load Python
module load Miniforge
eval "$('conda' 'shell.bash' 'hook' 2> /dev/null)"

conda activate diffseis
python run.py
