#!/bin/bash 
#SBATCH --partition=SCSEGPU_M1 
#SBATCH --qos=q_amsai 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=1G 
#SBATCH --job-name=task5
#SBATCH --output=output/task6/output_%x_%j.out
#SBATCH --error=output/task6/error_%x_%j.err 

module load anaconda3/23.5.2 
eval "$(conda shell.bash hook)" 
conda activate mobilenet 
export CUBLAS_WORKSPACE_CONFIG=:16:8

python /home/msai/xi0001ye/assignment-mobilenet-code/task6.py
