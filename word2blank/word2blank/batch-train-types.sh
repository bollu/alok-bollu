#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2____
#SBATCH --mem-per-cpu=8046
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --array=0-3

set -e 
set -o xtrace

DATE=`date '+%Y-%m-%d %H:%M:%S'`

METRICTYPES=(pseudoreimann euclid)
TRAINTYPES=(skipgramonehot cbow)

module add cuda/9.0
rm cur.model || true

METRICTYPE=${METRICTYPES[$(($SLURM_ARRAY_TASK_ID % 2))]}
TRAINTYPE=${TRAINTYPES[$(($SLURM_ARRAY_TASK_ID / 2))]}
FOLDERNAME=$SLURM_ARRAY_JOB_ID-$(git rev-parse HEAD)
mkdir -p models/$FOLDERNAME

NAME=$TRAINTYPE-$METRICTYPE

touch $FOLDERNAME/date-$DATE

./run.py train  --savepath models/$FOLDERNAME/$NAME.model --loadpath models/$FOLDERNAME/$NAME.model --metrictype $METRICTYPE --traintype $TRAINTYPE | tee models/$FOLDERNAME/$NAME.log
