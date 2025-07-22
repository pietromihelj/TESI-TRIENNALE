import argparse
import yaml
import os
from model import VAEEG
from Models_tr_aux_functions import train_VAEEG
from dataset import ClipDS
import torch

parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--yamal_file', type=str, required=True, help='configuers, path of .yaml fil')
parser.add_argument('--z_dim', type=int, required=True, help='z_dim')

opts = parser.parse_args()

with open(opts.yamal_file, 'r') as file:
    configs = yaml.safe_loads(file)

train_params = configs['Train']
model_params = configs['Model']
datset_params = configs['DataSet']
model_params['z_dim'] = opts.z_dim

if not os.path.isdir(train_params['model_dir']):
    os.makedirs(train_params['model_dir'])

model = VAEEG(in_channels=model_params['in_channels'],
              z_dim=model_params['z_dim'],
              negative_slope=model_params['negative_slope'],
              decoder_last_lstm=model_params['decoder_last_lstm'])

trainer = train_VAEEG(model, train_params['n_gpus'], train_params['ckpt_file'])
m_dir = os.path.join(train_params['model_dir'], "%s_z%d" % (os.path.basename(os.path.splitext(opts.yaml_file)[0]),opts.z_dim))

train_ds = ClipDS(data_dir=datset_params['data_dir'],
                  band_name=model_params['band_name'],
                  clip_len=datset_params['clip_len'])

train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=datset_params['batch_size'], drop_last=True, num_workers=0)
trainer.train(input_loader=train_loader, model_dir=m_dir, n_epochs=train_params['n_epochs'],lr=train_params['lr'],beta=train_params['beta'],n_print=train_params['n_print'])