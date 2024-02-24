#!/bin/bash
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

echo "Setting up bash environment"

source ~/.bashrc

set -e

CONDA_ENV_NAME=masters
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

export https_proxy=wwwcache.inf.ed.ac.uk:3128

cd /disk/scratch

if [ -d "s1950841" ]; then
    cd s1950841
    cd dataset/Namiki/drug/
else
    mkdir s1950841
    mkdir s1950841/dataset
    mkdir s1950841/dataset/Namiki
    mkdir s1950841/dataset/Namiki/drug
    cd s1950841/dataset/Namiki/drug
fi

rsync -vhz --partial --inplace /home/s1950841/MAEDDI/dataset/Namiki/drug/mols.sdf .

cd /disk/scratch/s1950841
rsync -vhz --partial --inplace /home/s1950841/MAEDDI/*.py .

python SSL_Data_Preprocess.py ./dataset/Namiki/drug/ conf ./dataset/Namiki/drug/mols.sdf conf
python SSL_Data_Preprocess.py ./dataset/Namiki/drug/ data ./dataset/Namiki/drug/mols.sdf mol

rsync -rvhz ./dataset/Namiki/drug/processed /home/s1950841/MAEDDI/dataset/Namiki/drug/

echo ""
echo "============="
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"