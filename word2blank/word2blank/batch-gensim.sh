#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2____
#SBATCH --mem-per-cpu=8046
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --array=0-1

set -e 
set -o xtrace

DATE=`date '+%Y-%m-%d--%H:%M:%S'`

module add cuda/9.0
rm cur.model || true

FOLDERNAME=$(git rev-parse --short HEAD)---$SLURM_ARRAY_JOB_ID---$DATE
mkdir -p models/$FOLDERNAME

NAME=gensim
touch models/$FOLDERNAME/date-$DATE

./train-gensim.py train  --savepath models/$FOLDERNAME/$NAME.model --loadpath models/$FOLDERNAME/$NAME.model | tee models/$FOLDERNAME/$NAME.log
