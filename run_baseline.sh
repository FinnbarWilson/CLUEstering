#!/bin/bash

# Normal CPU partition
#SBATCH -p COMPUTE

# Request 1 node
#SBATCH -N 1

# Python Multiprocessing Setup: 1 main task, but give it 16 cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Request 32 Gigabytes of RAM (2GB per core)
#SBATCH --mem=32G

# Set the time limit to the 12-hour default
#SBATCH --time=12:00:00

# Job name
#SBATCH -J CLUE_1Million

# Email Alerts 
#SBATCH --mail-user=finnbar.wilson.25@ucl.ac.uk
#SBATCH --mail-type=ALL

# Move to the working directory
cd /home/xucapfwi/CLUEstering/

echo "Job started at: $(date)"

# Initialise conda for bash and activate environment
source ~/.bashrc
conda activate CLUEstering

# Run the script
python validate_CLUE.py

echo "Job finished at: $(date)"
