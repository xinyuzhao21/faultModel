#!/bin/sh
#SBATCH -o out/experiment_%J.txt
#SBATCH -e out/experiment_%J.err
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
python -c 'import torch; print(torch.cuda.is_available())'
time python train.py  --config config.json

