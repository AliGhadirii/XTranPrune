#!/bin/bash
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=48GB
#PBS -l walltime=48:00:00
#PBS -m a
#PBS -l wd
#PBS -l jobfs=20GB

#PBS -P jk87

source /scratch/jk87/ag4694/miniconda3/bin/activate
module load cuda/11.7.0
module load cudnn/8.6.0-cuda11
conda activate FairPrune

# Set python path
export PYTHONPATH="/home/561/ag4694/Repos/FairPrune/:$PYTHONPATH"

# Call the script to run
python FairPrune.py --config Fitz17k_configs_NCI.yml