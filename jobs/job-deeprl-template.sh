#!/bin/bash

#SBATCH --job-name=train-{jobname}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --output slurm-{jobname}.log

TASKNAME={jobname}

ml load singularity

cd ~
singularity run --nv images/jkulhanek-deep-rl-pytorch-master-latest.simg train.py $TASKNAME
