#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1

python train.py -d 1 --config config.json
exit
