#!/bin/sh

# Slurm flags
#SBATCH -p ProdQ
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --job-name=PRQ-DT-1

# Charge job to myproject
#SBATCH -A tieng028c

# Write to file
#SBATCH -o prq-dt-1.txt

# Mail me on job start & end
#SBATCH --mail-user=marinara.marcato@tyndall.ie
#SBATCH --mail-type=BEGIN,END

cd $SLURM_SUBMIT_DIR

echo $GAUSS_SCRDIR

module load conda/2
source activate python3

python3 prq-dt-1.py