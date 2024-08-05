#!/bin/bash

python regular_deepdrug.py --configfile=./configs/TWOSIDES/deepdrug/deepdrug_random.yml >> deepdrug_ts_random_results.txt
#python regular_deepdrug.py --configfile=./configs/TWOSIDES/deepdrug/deepdrug_one_cls.yml >> deepdrug_ts_onecls_results.txt
python regular_deepdrug.py --configfile=./configs/BIOSNAP/deepdrug/deepdrug_random.yml >> deepdrug_bs_random_results.txt
#python regular_deepdrug.py --configfile=./configs/BIOSNAP/deepdrug/deepdrug_one_cls.yml >> deepdrug_bs_onecls_results.txt


#python clr_finetuning.py --configfile=./configs/DrugBank/clr/4_clr_allcls_RGCN.yml >> clr_rgcn_db_allcls_results.txt
#python clr_finetuning.py --configfile=./configs/DrugBank/clr/4_clr_allcls_RGIN.yml >> clr_rgin_db_allcls_results.txt
#python clr_finetuning.py --configfile=./configs/DrugBank/clr/4_clr_onecls_RGCN.yml >> clr_rgcn_db_onecls_results.txt
#python clr_finetuning.py --configfile=./configs/DrugBank/clr/4_clr_onecls_RGIN.yml >> clr_rgin_db_onecls_results.txt
#python clr_finetuning.py --configfile=./configs/TWOSIDES/clr/4_clr_allcls_RGCN.yml >> clr_rgcn_ts_allcls_results.txt
#python clr_finetuning.py --configfile=./configs/TWOSIDES/clr/4_clr_allcls_RGIN.yml >> clr_rgin_ts_allcls_results.txt
#python clr_finetuning.py --configfile=./configs/TWOSIDES/clr/4_clr_onecls_RGCN.yml >> clr_rgcn_ts_onecls_results.txt
#python clr_finetuning.py --configfile=./configs/TWOSIDES/clr/4_clr_onecls_RGIN.yml >> clr_rgin_ts_onecls_results.txt
#python clr_finetuning.py --configfile=./configs/BIOSNAP/clr/4_clr_allcls_RGCN.yml >> clr_rgcn_bs_allcls_results.txt
#python clr_finetuning.py --configfile=./configs/BIOSNAP/clr/4_clr_allcls_RGIN.yml >> clr_rgin_bs_allcls_results.txt
#python clr_finetuning.py --configfile=./configs/BIOSNAP/clr/4_clr_onecls_RGCN.yml >> clr_rgcn_bs_onecls_results.txt
#python clr_finetuning.py --configfile=./configs/BIOSNAP/clr/4_clr_onecls_RGIN.yml >> clr_rgin_bs_onecls_results.txt
