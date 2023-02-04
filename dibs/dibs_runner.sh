#!/bin/bash

num_variables=$1

default_path='er1-lingauss-d050:v1'
data_path=${2:-$default_path}
def_time='3:00:00'
time=${3:-$def_time}

seeds=(20 23)
array_len=$(( ${#seeds[@]} ))
echo $array_len
echo $score
output_file="dibs_out/${data_path}_seed_%a-%A.out"

command="sbatch --array=1-${array_len}%512 --job-name dibs:${data_path} --output ${output_file} --time ${time} dibs_job.sh ${num_variables} ${data_path}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }

echo "Job ID"" ""${job_id}"" -> ""${data_path}" 
echo ""