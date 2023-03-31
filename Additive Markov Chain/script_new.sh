#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=23:59:00
#SBATCH --nodelist=komputasi09

srun python3 recommendation.py
