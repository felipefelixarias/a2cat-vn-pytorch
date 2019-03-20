#/bin/bash

#SBATCH --job-name=chouse-single-goal-hard-bignet
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1

TASKNAME=chouse-single-goal-hard-bignet

ml load singularity
singularity exec images/jkulhanek
singularity exec --nv images/jkulhanek-target-driven-visual-navigation-master-latest.simg python3 ~/experiments/target-driven-visual-navigation/train.py $TASKNAME