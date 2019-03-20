#!/bin/bash
cat ~/experiments/target-driven-visual-navigation/jobs/job-template.sh | sed "s/{jobname}/$1/g" | tee job.sh > /dev/null
sbatch job.sh
rm job.sh