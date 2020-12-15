#!/bin/sh

#SBATCH --ntasks=1
#SBATCH -t 04:00:00               # max runtime is 4 hours
#SBATCH -J  tensorboard_server    # name
#SBATCH -o  tb-%J.out #TODO: Where to save your output

LOG_DIR= saved/log/CIFAR_10_ResNet_18_BASE/ #TODO: Your TF model directory

let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

tensorboard --logdir="${MODEL_DIR}" --LOG_DIR=$ipnport