#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=word2____
#SBATCH --mem-per-cpu=8046
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END

module add cuda/9.0
rm cur.model || true
./run.py train 
