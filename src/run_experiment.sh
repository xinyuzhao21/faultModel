#!/bin/sh
#SBATCH -o my_output.txt
#SBATCH -e my_errors.txt
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

python train.py  --config config.json

