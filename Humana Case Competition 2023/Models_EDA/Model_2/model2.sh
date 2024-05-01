#!/bin/bash
#SBATCH --job-name=train2_xgboost 
#SBATCH --output=train2_xgboost_%j.out  # Output file
#SBATCH --error=train2_xgboost_%j.err   # Error file
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=abdullah.kazi@sjsu.edu
#SBATCH --mem=16G

module load python3
python3 model2.py
