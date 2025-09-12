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
source /opt/spack/opt/spack/linux-rocky9-x86_64/gcc-13.2.0/miniconda3-22.11.1-tn534fvb4uy4wrf7m2zcwpiycdzlebd6/bin/activate my_tesy_env
cd /u/pmihelj/TESI-TRIENNALE
python3 pre_process_and_dataset_creation.py \
--input_edf_dir "/u/pmihelj/nchsdb/sleep_data"\
--output_raw_data_dir "/u/pmihelj/raw_datas/sleep_raw"\
--output_dataset_dir "/u/pmihelj/datasets/sleep_dataset"\
--log_file_path "/u/pmihelj/raw_datas/sleep_raw/log.csv"\
--num_cpus "$SLURM_CPUS_PER_TASK"
