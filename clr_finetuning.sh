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
python clr_finetuning.py --configfile=./configs/mae_clr_exps/clr_config.yml

echo ""
echo "============="
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
