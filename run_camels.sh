#!/bin/bash

#SBATCH --job-name=wavelet_camels
#SBATCH --ntasks=1
#SBATCH --time=00-16:00
#SBATCH --mem=6000M
#SBATCH --gpus-per-node=1
#SBATCH --array=1-621

module purge
module load python/3.10
module load r/4.2.1
source /home/jswyou/projects/def-quiltyjo/jswyou/venv/bin/activate

echo "Starting run at: `date`"

echo "Starting task $SLURM_ARRAY_TASK_ID"

CSV_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" csv_filenames.txt)
python main.py --csv_file $CSV_FILE

echo "Program finished with exit code $? at: `date`"
