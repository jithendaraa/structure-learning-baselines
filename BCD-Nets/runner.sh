#!/bin/bash

num_variables=$1
num_steps=$2

default_path='sbm4-lingauss-d020-c001:v1'
data_path=${3:-$default_path}
def_time='9:00:00'
time=${4:-$def_time}

seeds=(0 15 18 19 20 21 22 23 24)
array_len=$(( ${#seeds[@]} ))
echo $array_len
echo $score
output_file="slurm_runs/${data_path}_seed_%a-%A.out"

command="sbatch --array=1-${array_len}%512 --job-name bcd:${data_path} --output ${output_file} --time ${time} bcd_job.sh ${num_variables} ${data_path} ${num_steps}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }

echo "Job ID"" ""${job_id}"" -> ""${data_path}" 
echo ""