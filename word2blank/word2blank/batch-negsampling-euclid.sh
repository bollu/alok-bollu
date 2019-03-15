#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2____
#SBATCH --mem-per-cpu=32000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END

set -e 
set -o xtrace
METRICTYPE=euclid
TRAINTYPE=skipgramnegsampling
EPOCHS=1
BATCHSIZE=64
EMBEDSIZE=200
LEARNINGRATE=0.05
WINDOWSIZE=4
DATE=`date '+%Y-%m-%d--%H:%M:%S'`

module add cuda/9.0
rm cur.model || true

FOLDERNAME=$(git rev-parse --short HEAD)---$SLURM_ARRAY_JOB_ID---$DATE
mkdir -p models/$FOLDERNAME

NAME=$TRAINTYPE-$METRICTYPE
touch models/$FOLDERNAME/date-$DATE

./run.py train  \
        --savepath models/$FOLDERNAME/$NAME.model \
        --loadpath models/$FOLDERNAME/$NAME.model \
        --epochs $EPOCHS \
        --batchsize $BATCHSIZE \
        --embedsize $EMBEDSIZE \
        --learningrate $LEARNINGRATE \
        --windowsize $WINDOWSIZE \
        --metrictype $METRICTYPE \
        --traintype $TRAINTYPE | tee models/$FOLDERNAME/$NAME.log
