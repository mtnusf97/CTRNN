#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --account=def-cannoj9
source ../.venv/bin/activate
echo 'Hello, world!'
python3 /home/mtnusf97/projects/def-cannoj9/mtnusf97/CTRNN/dataset_generator/gaussian_loss_derivative_generator.py
echo 'Done'
sleep 5