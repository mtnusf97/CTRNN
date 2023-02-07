#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-cannoj9
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2G

source ../.venv/bin/activate
echo 'starting point'
python3 /home/mtnusf97/projects/def-cannoj9/mtnusf97/CTRNN/run_exp.py -c /home/mtnusf97/projects/def-cannoj9/mtnusf97/CTRNN/config/gaussian_derivative_loss.yaml
echo 'Done'
sleep 5