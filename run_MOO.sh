#!/bin/sh
#SBATCH -o ./rerun_moo_pam/moo_plus48_%j.output
#SBATCH -p DGXq
#SBATCH -n 1

export PATH=/home/yuhao001/anaconda3/envs/mos2_project/bin:$PATH
cd /home/yuhao001/projects/MOO/
 
#time python main.py  --config ./configs/debug_config_moo_real.json --mode real  
#time python main.py  --config ./configs/debug_config_moo_ad_linear.json --mode synthetic
#time python compare.py 

time python ./run_scripts/moo_design_loops.py  --config ./configs/moo_pam_real_plus48.json --mode real
#moo_pam_real_plus48.json