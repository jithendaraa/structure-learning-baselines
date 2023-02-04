#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

num_nodes=$1
datapath=$2

seeds=(20 23)

num_steps=3000
particles=1000

array_len=$(( ${#exp_edges[@]} * ${#num_steps[@]} ))

seed=${seeds[ $((  (${SLURM_ARRAY_TASK_ID} )  % ${#seeds[@]} )) ]}
start=`date +%s`
echo "Script"

module load anaconda/3
conda activate dibs_env
echo `date` "Python starting"
echo "python dibs_exp.py -s ${seed} -d ${num_nodes} --num_steps ${num_steps} --particles ${particles} --data_path ${datapath}"
python dibs_exp.py -s ${seed} -d ${num_nodes} --num_steps ${num_steps} --particles ${particles} --data_path ${datapath}

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"