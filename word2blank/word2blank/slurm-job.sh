#!/bin/bash
#SBATCH -p long
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2blank
#SBATCH --mem-per-cpu=4096
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END

module add cuda/8.0
python run.py train --savepath cur.model --loadpath cur.model
