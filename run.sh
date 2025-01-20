#!/bin/bash

#PBS -N diffseis_64x256_200k
#PBS -q gpu
#PBS -b 1
#PBS -r n
#PBS -l elapstim_req=36:00:00
#PBS -o stdout.%s.%j
#PBS -e stderr.%s.%j
#PBS --custom gpusetnum-lhost=1

cd $PBS_O_WORKDIR
module load Miniforge/24.3.0
eval "$('conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate diffseis-cuda

#module load Python
#module load CUDA cuDNN

#module load python torch


python run.py 
