#!/bin/bash
#
#SBATCH --job-name=VAEEG_training
#SBATCH --partition=turing-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
source /opt/spack/opt/spack/linux-rocky9-x86_64/gcc-13.2.0/miniconda3-22.11.1-tn534fvb4uy4wrf7m2zcwpiycdzlebd6/bin/activate my_tesy_env
cd /u/pmihelj/TESI-TRIENNALE
python3 -u model_training.py \
--model_dir "/u/pmihelj/models" \
--in_channels 1 \
--z_dim 8 \
--negative_slope 0.2 \
--decoder_last_lstm \
--ckpt_file " /u/pmihelj/models/VAEEG_delta_z8/ckpt_epoch_9_660000.ckpt" \
--data_dir "/u/pmihelj/datasets/Training_dataset/train" \
--band_name "delta" \
--clip_len 250 \
--batch_size 16 \
--n_epochs 100 \
--lr 0.001 \
--beta 0.0001 \
--n_print 50000 \
--n_gpus=0
