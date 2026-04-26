#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=PIF_over_time
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=0-49%20

module purge
module load anaconda
export PYTHONWARNINGS="ignore"

conda activate /proj/characklab/projects/kieranf/flood_damage_index/fli-env-v1
python3.12 policies_in_force.py