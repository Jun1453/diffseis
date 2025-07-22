#!/bin/bash

#PBS -N diffseis_64x256_200ep
#PBS -q gpu
#PBS -b 1
#PBS -r n
#PBS -l elapstim_req=24:00:00
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


#python run.py
#python train.py
#export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_DEBUG=INFO
#export NCCL_P2P_DISABLE=0
#export NCCL_P2P_LEVEL=NVL
accelerate launch --num_processes=1 train.py
