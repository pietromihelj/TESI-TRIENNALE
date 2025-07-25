import argparse
import os
from model import VAEEG
from Models_tr_aux_functions import train_VAEEG
from dataset import ClipDS
import torch

parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--model_dir', type=str, required=True, help='model_dir')
parser.add_argument('--in_channels', type=int, required=True, help='in_channels')
parser.add_argument('--z_dim', type=int, required=True, help='z_dim')
parser.add_argument('--negative_slope', type=float, required=False, help='negative_slope', default=0.2)
parser.add_argument('--decoder_last_lstm', type=bool, required=False, help='decoder_last_lstm', default=True)
parser.add_argument('--n_gpus', type=int, required=False, help='n_gpus', default=0)
parser.add_argument('--ckpt_file', type=str, required=False, help='ckpt_file', default=None)
parser.add_argument('--data_dir', type=str, required=True, help='data_dir')
parser.add_argument('--band_name', type=str, required=True, help='band_name')
parser.add_argument('--clip_len', type=int, required=False, help='clip_len', default=250)
parser.add_argument('--batch_size', type=int, required=True, help='batch_size')
parser.add_argument('--n_epochs', type=int, required=True, help='n_epochs')
parser.add_argument('--lr', type=float, required=True, help='lr')
parser.add_argument('--beta', type=float, required=True, help='beta')
parser.add_argument('--n_print', type=int, required=True, help='n_print')

opts = parser.parse_args()

model_dir = opts.model_dir
in_channels = opts.in_channels
z_dim = opts.z_dim
negative_slope = opts.negative_slope
decoder_last_lstm = opts.decoder_last_lstm
n_gpus = opts.n_gpus
ckpt_file = opts.ckpt_file
data_dir = opts.data_dir
band_name = opts.band_name
clip_len = opts.clip_len
batch_size = opts.batch_size
n_epochs = opts.n_epochs
lr = opts.lr
beta = opts.beta
n_print = opts.n_print

print('DEBUG: tutte le variabili assegnate')

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

model = VAEEG(in_channels=in_channels,
              z_dim=z_dim,
              negative_slope=negative_slope,
              decoder_last_lstm=decoder_last_lstm)

print('DEBUG: modello creato')

trainer = train_VAEEG(model, n_gpus=n_gpus, ckpt_file=ckpt_file)
m_dir = os.path.join(model_dir, "%s_%s_z%d" % ('VAEEG','Band_name',opts.z_dim))

print('DEBUG: trainer creato')

train_ds = ClipDS(data_dir=data_dir,
                  band_name=band_name,
                  clip_len=clip_len)

print('DEBUG: dataset creato')

train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0)

print('DEBUG: train loader creato')
print('DEBUG: inizio training')
trainer.train(input_loader=train_loader, model_dir=m_dir, n_epochs=n_epochs,lr=lr,beta=beta,n_print=n_print)
