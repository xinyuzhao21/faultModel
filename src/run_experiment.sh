#!/bin/sh
#SBATCH -o out/experiment_%J.txt
#SBATCH -e out/experiment_%J.err
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
echo `pwd`
echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module unload cudnn/4.0
#module unload cudnn/5.1
echo 'what1'
echo 'what2'
__conda_setup="$('/home/xinyuzhao/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
echo 'what3'
if [ $? -eq 0 ]; then
	eval "$__conda_setup"
else
	if [ -f "/home/xinyuzhao/anaconda3/etc/profile.d/conda.sh" ]; then
		. "/home/xinyuzhao/anaconda3/etc/profile.d/conda.sh"
	else
		export PATH="/home/xinyuzhao/anaconda3/bin:$PATH"
	fi  
fi
unset __conda_setup
# <<< conda initialize <<
conda init bash
sleep 1
conda activate faultModel
sleep 1
python -c 'import torch; print(torch.cuda.is_available())'
time python train.py  --config config.json

