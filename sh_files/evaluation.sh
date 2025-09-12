#!/bin/bash
#
#SBATCH --job-name=VAEEG_training
#SBATCH --partition=turing-wide
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

source /opt/spack/opt/spack/linux-rocky9-x86_64/gcc-13.2.0/miniconda3-22.11.1-tn534fvb4uy4wrf7m2zcwpiycdzlebd6/bin/activate my_tesy_env
cd /u/pmihelj/TESI-TRIENNALE

python3 -u eval.py \
--data_dir "/u/pmihelj/datasets/eval" \
--model VAEEG \
--model_save_name VAEEG \
--model_save_files  /u/pmihelj/models/models/VAEEG/delta_band.ckpt /u/pmihelj/models/models/VAEEG/theta_band.ckpt /u/pmihelj/models/models/VAEEG/alpha_band.ckpt /u/pmihelj/models/models/VAEEG/low_beta_band.ckpt /u/pmihelj/models/models/VAEEG/high_beta_band.ckpt\
--params 8 10 12 10 10  \
--out_dir /u/pmihelj/results 