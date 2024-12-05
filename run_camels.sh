#!/bin/bash

#SBATCH --job-name=wavelet_camels
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=12000M
#SBATCH --gpus-per-node=1
#SBATCH --array=1-621

module purge
module load python/3.10
module load r/4.4
source /home/jswyou/projects/def-quiltyjo/jswyou/tf/bin/activate
pip install tensorflow --upgrade
pip install keras --upgrade
pip install rpy2 --upgrade

echo "Starting run at: `date`"

echo "Starting task $SLURM_ARRAY_TASK_ID"

CSV_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" csv_filenames.txt)
python main.py --csv_filename $CSV_FILE

echo "Program finished with exit code $? at: `date`"
