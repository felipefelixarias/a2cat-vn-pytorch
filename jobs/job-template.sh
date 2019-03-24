#!/bin/bash

#SBATCH --job-name=train-{jobname}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --output slurm-{jobname}.log

TASKNAME={jobname}

ml load singularity
SCRATCH_DIRECTORY=/lscratch/${USER}/${SLURM_JOBID}.stallo-adm.uit.no
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}
git init
git remote add origin git@github.com:jkulhanek/target-driven-visual-navigation.git
git pull origin master
#singularity pull shub://jkulhanek/target-driven-visual-navigation:latest

cd ~
singularity run -B ${SCRATCH_DIRECTORY}:/experiment --nv images/jkulhanek-target-driven-visual-navigation-master-latest.simg train.py $TASKNAME
