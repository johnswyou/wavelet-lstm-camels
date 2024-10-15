#!/bin/bash

#SBATCH --job-name=vanilla_camels
#SBATCH --ntasks=1
#SBATCH --time=00-36:00
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=v100l:1
#SBATCH --array=1,3

module purge
module load r/4.2.1
source /home/jswyou/projects/def-quiltyjo/jswyou/venv/bin/activate

echo "Starting run at: `date`"

echo "Starting task $SLURM_ARRAY_TASK_ID"

python main_vanilla.py --leadtime $SLURM_ARRAY_TASK_ID

echo "Program finished with exit code $? at: `date`"
