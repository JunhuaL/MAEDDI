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
    if [ -d "dataset" ]; then
        cd dataset
        if [ -d "Namiki" ]; then
            cd Namiki
        else
            mkdir Namiki
            mkdir Namiki/drug/
			mkdir Namiki/drug/processed
            cd Namiki/drug/
        fi
    else
        mkdir dataset
        mkdir dataset/Namiki/
        mkdir dataset/Namiki/drug/
		mkdir dataset/Namiki/drug/processed
        cd dataset/Namiki/drug/
    fi
else
    mkdir s1950841
    mkdir s1950841/dataset
    mkdir s1950841/dataset/Namiki
    mkdir s1950841/dataset/Namiki/drug
	mkdir s1950841/dataset/Namiki/drug/processed
    cd s1950841/dataset/Namiki/drug
fi

rsync -rvhz --partial --inplace /home/s1950841/MAEDDI/dataset/Namiki/drug/processed .
cd /disk/scratch/s1950841
rsync -vhz --partial --inplace /home/s1950841/MAEDDI/*.py .

python molconf_pretraining.py 20 4 random mae

rsync -rvhz ./model_checkpoints /home/s1950841/MAEDDI/

echo ""
echo "============="
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
