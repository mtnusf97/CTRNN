#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --account=def-cannoj9
echo 'Hello, world!'
source ../.venv/bin/activate
python3 first_job_script.py
echo 'Bye, world!'
sleep 30