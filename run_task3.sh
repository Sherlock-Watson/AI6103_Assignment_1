#!/bin/bash 
#SBATCH --partition=SCSEGPU_M2 
#SBATCH --qos=q_dmsai 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=1G 
#SBATCH --job-name=task3
#SBATCH --output=output/%x/output_%x_%j.out
#SBATCH --error=output/%x/error_%x_%j.err 

module load anaconda3/23.5.2 
eval "$(conda shell.bash hook)" 
conda activate mobilenet 
export CUBLAS_WORKSPACE_CONFIG=:16:8

python /home/msai/xi0001ye/assignment-mobilenet-code/task3.py
