#!/bin/bash
cat ~/experiments/target-driven-visual-navigation/jobs/job-deeprl-template.sh | sed "s/{jobname}/$1/g" | tee job.sh > /dev/null
sbatch job.sh
rm job.sh