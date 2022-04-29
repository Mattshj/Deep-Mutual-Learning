#!/bin/bash
#SBATCH --account=def-dhadidi
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M               # memory (per node)
#SBATCH --time=0-0:30            # time (DD-HH:MM)
#SBATCH --mail-user=h.fazli.k@gmail.com
#SBATCH --mail-type=ALL
python format_and_convert_data_train.py