#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2____
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --array=0

set -e 
set -o xtrace
METRICTYPES=(euclid)
DATE=`date '+%Y-%m-%d--%H:%M:%S'`

module add cuda/9.0
rm cur.model || true

NMETRICTYPES=${#METRICTYPES[@]}

METRICTYPE=${METRICTYPES[$(($SLURM_ARRAY_TASK_ID % $NMETRICTYPES))]}
TRAINTYPE=skipgramnegsampling
FOLDERNAME=$(git rev-parse --short HEAD)---$SLURM_ARRAY_JOB_ID---$DATE
mkdir -p models/$FOLDERNAME

NAME=$TRAINTYPE-$METRICTYPE
touch models/$FOLDERNAME/date-$DATE

./run.py train  --savepath models/$FOLDERNAME/$NAME.model --loadpath models/$FOLDERNAME/$NAME.model --metrictype $METRICTYPE --traintype $TRAINTYPE | tee models/$FOLDERNAME/$NAME.log
