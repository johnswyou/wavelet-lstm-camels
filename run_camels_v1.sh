#!/bin/bash

#SBATCH --job-name=wavelet_camels
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=12000M
#SBATCH --gpus-per-node=1
#SBATCH --array=1-167

module purge
module load python/3.11
module load r/4.4              # R 4.4 now on PATH

# ── R packages needed by hydroIVS ───────────────────────────────────────────────
# Install dependencies directly from CRAN mirror
Rscript -e 'install.packages(c("RANN","Boruta","RRF"), repos = "https://cloud.r-project.org", quiet = TRUE)'

# Fetch hydroIVS source and install it with R CMD INSTALL
git clone --depth 1 "https://github.com/johnswyou/hydroIVS.git"              \
        $SLURM_TMPDIR/hydroIVS
R CMD INSTALL --no-multiarch --with-keep.source $SLURM_TMPDIR/hydroIVS     \
        >/dev/null

# ── Python virtual‑env ──────────────────────────────────────────────────────────
virtualenv --no-download $SLURM_TMPDIR/tf
source $SLURM_TMPDIR/tf/bin/activate
pip install --no-index --upgrade pip
pip install --no-index matplotlib
pip install --no-index tensorflow
pip install --no-index keras
pip install --no-index rpy2==3.1.0
pip install --no-index scikit-learn
pip install --no-index scipy
pip install --no-index pandas==1.5.3
# pip install tensorflow --upgrade
# pip install keras --upgrade

# ── Job execution ───────────────────────────────────────────────────────────────
echo "Starting run at: $(date)"
echo "Starting task $SLURM_ARRAY_TASK_ID"

CSV_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" csv_filenames_v1.txt)
python main.py --csv_filename "$CSV_FILE" --verbose

echo "Program finished with exit code $? at: $(date)"
