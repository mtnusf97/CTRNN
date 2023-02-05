#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --account=def-cannoj9
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2G
nvidia-smi