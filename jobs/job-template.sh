#!/bin/bash

#SBATCH --job-name=chouse-single-goal-hard-bignet
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=12G
#SBATCH --gres=gpu:1

TASKNAME=chouse-single-goal-hard-bignet

ml load singularity
SCRATCH_DIRECTORY=/lscratch/${USER}/${SLURM_JOBID}.stallo-adm.uit.no
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}
git init
git remote add origin git@github.com:jkulhanek/target-driven-visual-navigation.git
git pull origin master

cd ~
singularity exec images/jkulhanek
singularity exec -B ${SCRATCH_DIRECTORY}:/experiment --nv images/jkulhanek-target-driven-visual-navigation-master-latest.simg python3 ~/experiments/target-driven-visual-navigation/train.py $TASKNAME