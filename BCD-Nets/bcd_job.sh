#!/bin/bash
#SBATCH --time=27:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

num_variables=$1
data_path=$2
num_steps=$3


seeds=(0 15 18 19 20 21 22 23 24)
seed=${seeds[ $((  (${SLURM_ARRAY_TASK_ID})  % ${#seeds[@]} )) ]}

module load anaconda/3
conda activate mybcd

start=`date +%s`
echo "Begin run"
echo "python main.py -s ${seed} --dim ${num_variables} --num_steps ${num_steps} --do_ev_noise --sem_type linear-gauss --batch_size 256 --use_my_data True --use_wandb --data_path ${data_path}"

python main.py -s ${seed} --dim ${num_variables} --num_steps ${num_steps} --do_ev_noise --sem_type linear-gauss --batch_size 256 --use_my_data True --use_wandb --data_path ${data_path}

end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"
