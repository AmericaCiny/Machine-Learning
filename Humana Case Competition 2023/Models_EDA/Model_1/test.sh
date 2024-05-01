#!/bin/bash
#SBATCH --job-name=test 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
#SBATCH --mail-user=abdullah.kazi@sjsu.edu

python3 test1.py
