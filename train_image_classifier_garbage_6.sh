#!/bin/bash
#SBATCH --account=def-dhadidi
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M               # memory (per node)
#SBATCH --time=0-1:00            # time (DD-HH:MM)
#SBATCH --mail-user=h.fazli.k@gmail.com
#SBATCH --mail-type=ALL
python train_image_classifier_garbage_6.py