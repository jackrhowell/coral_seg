#!/bin/bash
#See https://slurm.schedmd.com/job_array.html
#See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

SBATCH --partition=gpu #gpu,kill-shared #sadow, gpu, shared, kill-shared, exclusive-long

SBATCH --gres=gpu:1
SBATCH --cpus-per-task=4 ## cores per task
SBATCH --mem=36gb ## max amount of memory per node you require
SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow

SBATCH --job-name=coral_seg
SBATCH --output=job.out
SBATCH --output=./logs/slurm_output/coral-trainer-%A.out
SBATCH --mail-type=START,END,FAIL
SBATCH --mail-user=jrhowell@hawaii.edu

# Load python profile and activate environment
## source ~/profiles/auto.profile
module load lang/Anaconda3
source activate coral_seg_v2

# use this command to run a python script
python evaluate_gp.py 

# use this command to run an ipynb and save outputs in the notebook
# jupyter nbconvert --execute --clear-output file.ipynb 

# Another command to create a .py script, then run that from a ipynb:
# jupyter nbconvert file.ipynb --to python
# python file.py

echo "This is job $SLURM_ARRAY_JOB_ID task running on $HOSTNAME"

python train.py