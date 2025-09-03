#!/bin/bash
#
#SBATCH --job-name=VAEEG_training
#SBATCH --partition=turing_wide
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

source /opt/spack/opt/spack/linux-rocky9-x86_64/gcc-13.2.0/miniconda3-22.11.1-tn534fvb4uy4wrf7m2zcwpiycdzlebd6/bin/activate my_tesy_env
cd /u/pmihelj/TESI-TRIENNALE

python3 -u eval.py \
--data_dir \
--models \
--model_save_files\
--params \
--out_dir \