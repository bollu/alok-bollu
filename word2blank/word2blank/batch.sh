#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2____
#SBATCH --mem-per-cpu=8046
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --array=0-2

set -e 
set -o xtrace

TYPES=(euclid reimann pseudoreimann)

module add cuda/9.0
rm cur.model || true

FOLDERNAME=$(git rev-parse HEAD)
mkdir -p models/$FOLDERNAME
TYPE=${TYPES[$SLURM_ARRAY_TASK_ID]}

./run.py train  --savepath models/$FOLDERNAME/$TYPE.model $TYPE
