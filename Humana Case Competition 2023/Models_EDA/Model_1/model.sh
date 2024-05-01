#!/bin/bash
#SBATCH --job-name=train_xgboost 
#SBATCH --output=train_xgboost_%j.out  # Output file
#SBATCH --error=train_xgboost_%j.err   # Error file
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=abdullah.kazi@sjsu.edu
#SBATCH --mem=16G

python3 model.py
