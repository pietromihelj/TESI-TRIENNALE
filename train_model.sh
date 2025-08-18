#!/bin/bash
#
#SBATCH --job-name=VAEEG_training
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --output=slurms/VAEEG/ottavo_training/slurm-%j.out
#SBATCH --error=slurms/VAEEG/ottavo_training/slurm-%j.err
#SBATCH --gres=gpu:1

source /opt/spack/opt/spack/linux-rocky9-x86_64/gcc-13.2.0/miniconda3-22.11.1-tn534fvb4uy4wrf7m2zcwpiycdzlebd6/bin/activate my_tesy_env
cd /u/pmihelj/TESI-TRIENNALE
python3 -u model_training.py \
--model_dir "/u/pmihelj/models" \
--in_channels 1 \
--z_dim 10 \
--negative_slope 0.2 \
--decoder_last_lstm \
--ckpt_file "/u/pmihelj/models/VAEEG_high_beta_z10/ckpt_epoch_941_38200000.ckpt" \
--data_dir "/u/pmihelj/datasets/Training_dataset/train" \
--band_name "high_beta" \
--clip_len 250 \
--batch_size 16 \
--n_epochs 200 \
--lr 0.0001 \
--beta 0.0001 \
--n_print 300000 \
--n_gpus=1
