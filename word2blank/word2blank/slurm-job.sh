#!/bin/bash
#SBATCH -p long
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2____
#SBATCH --mem-per-cpu=4096
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END

module add cuda/8.0
rm cur.model || true
python run.py train --savepath no-neg-sample.model
