#!/bin/bash
#
#SBATCH --job-name=pre_processing
#SBATCH --partition=turing-wide
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=1-23:00:00
#
#SBATCH --output=slurm-wide-%j.out
#SBATCH --error=slurm-wide-%j.err
#
#CONDA_BASE_ACTIVATE_SCRIPT="/opt/spack/opt/spack/linux-rocky9-x86_64/gcc-13.2.0/miniconda3-22.11.1-tn534fvb4uy4wrf7m2zcwpiycdzlebd6/bin/activate"
#ENV_NAME="my_tesy_env"
source "$CONDA_BASE_ACTIVATE_SCRIPT" "$ENV_NAME"
cd /u/pmihelj/TESI-TRIENNALE
python3 pre_process_and_dataset_creation.py