#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2____
#SBATCH --mem-per-cpu=8046
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --array=0-5

set -e 
set -o xtrace

METRICTYPES=(euclid pseudoreimann)
TRAINTYPES=(skipgram cbow)

module add cuda/9.0
rm cur.model || true

mkdir -p models/$FOLDERNAME
METRICTYPE=${METRICTYPES[$(($SLURM_ARRAY_TASK_ID % 3))]}
TRAINTYPE=${TRAINTYPES[$(($SLURM_ARRAY_TASK_ID / 3))]}
FOLDERNAME=$(git rev-parse HEAD)

NAME=$TRAINTYPE-$METRICTYPE

./run.py train  --savepath models/$FOLDERNAME/$NAME.model --loadpath models/$FOLDERNAME/$NAME.model --metrictype $METRICTYPE --traintype $TRAINTYPE | tee models/$FOLDERNAME/$NAME.log
