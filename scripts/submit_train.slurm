#!/bin/bash
##See https://slurm.schedmd.com/job_array.html
##See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=gpu #gpu,kill-shared #sadow, gpu, shared, kill-shared, exclusive-long

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 ## cores per task
#SBATCH --mem=36gb ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long

#SBATCH --job-name=coral_seg
#SBATCH --output=./logs/slurm_output/coral-trainer-%j.out
#SBATCH --error=./logs/slurm_output/coral-trainer-%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=jrhowell@hawaii.edu

# Load python profile and activate environment
module load lang/Anaconda3
module load system/CUDA/12.2.0
source activate coral_seg_v2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Host: $HOSTNAME"
echo "PWD:  $(pwd)"
echo "Python: $(which python)"
python -V
conda info -e
nvidia-smi
nvcc --version

echo "This is job $SLURM_ARRAY_JOB_ID task running on $HOSTNAME"

python train.py