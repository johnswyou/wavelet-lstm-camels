#!/bin/bash
#SBATCH --job-name=wavelet_camels
#SBATCH --time=15:59:00
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --gpus-per-node=1
#SBATCH --array=1-6

###############################################################################
# 1.  Modules                                                                 #
###############################################################################
module purge
module load gcc/12.3          # same compiler you will use when you build pkgs
module load python/3.11
module load r/4.4

###############################################################################
# 2.  R packages (all from local *.tar.gz files, installed in dependency order)
###############################################################################
export PKGSRC=/project/6040293/jswyou/pkgsrc          # directory with tarballs
export R_LIBS_USER=$SLURM_TMPDIR/R_libs
mkdir -p "$R_LIBS_USER"

# Install once per node / array‑task‑group
if [ ! -d "$R_LIBS_USER/hydroIVS" ]; then
    ordered=(Rcpp RcppEigen ranger Boruta RANN RRF hydroIVS)
    for pkg in "${ordered[@]}"; do
        tar=$(ls "$PKGSRC/${pkg}"_*.tar.gz | head -n 1)
        if [ -f "$tar" ]; then
            echo "Installing $pkg from $(basename "$tar")"
            R CMD INSTALL --no-multiarch --library="$R_LIBS_USER" "$tar"
        else
            echo "ERROR: $pkg tarball not found in $PKGSRC" >&2
            exit 1
        fi
    done
fi

###############################################################################
# 3.  Python environment                                                      #
###############################################################################
virtualenv --no-download "$SLURM_TMPDIR/tf"
source "$SLURM_TMPDIR/tf/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index \
    matplotlib \
    tensorflow \
    keras \
    rpy2==3.1.0 \
    scikit-learn \
    scipy \
    pandas==1.5.3

###############################################################################
# 4.  Run your workload                                                       #
###############################################################################
echo "Starting run at: $(date)"
echo "Starting task $SLURM_ARRAY_TASK_ID"

CSV_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" csv_filenames_v5.txt)
python main.py --csv_filename "$CSV_FILE" --verbose

echo "Program finished with exit code $? at: $(date)"
