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
	cd dataset/Chemberta/drug/
else
    mkdir s1950841
    cd s1950841

fi

mkdir dataset
cd dataset
mkdir Chemberta
cd Chemberta
mkdir drug
cd drug
mkdir processed

rsync -vhz --partial --inplace /home/s1950841/MAEDDI/dataset/Chemberta/drug/drug.csv .
rsync -rvhz --partial --inplace /home/s1950841/MAEDDI/dataset/Chemberta/drug/processed .
cd /disk/scratch/s1950841
rsync -vhz --partial --inplace /home/s1950841/MAEDDI/*.py .


python clr_pretraining.py 20 4 random

rsync -rvhz ./model_checkpoints /home/s1950841/MAEDDI/

echo ""
echo "============="
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"