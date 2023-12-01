#!/bin/bash
#SBATCH --partition=p100
#SBATCH -J pytorchjob
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
bash scripts/gnmt8/xpipe_throughput.sh
sleep 2
bash scripts/gnmt16/xpipe_throughput.sh
