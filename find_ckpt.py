from tensorboard.backend.event_processing import event_accumulator
import argparse
import torch
from utils import check_type, get_abs_path
import numpy as np
import os
from collections import defaultdict
# directory dei checkpoint
parser = argparse.ArgumentParser(description='Find checkpoint')
parser.add_argument('--ckpt_dir', type=str, required=True, help='dir')
parser.add_argument('--events_dir', type=str, required=True, help='dir')
opts = parser.parse_args()
ckpt_dir = opts.ckpt_dir
events_dir = opts.events_dir

# lista tutti i file ckpt
def get_path_list(d_path, f_extensions=None, sub_d=False):
    """
    d_path: path assoluto della directory
    f_extensions: estensioni permesse (lista di stringhe, es. ['.txt','.csv'])
                  se None, accetta tutte le estensioni
    sub_d: ricorsività nelle sottodirectory
    """
    check_type('d_path', d_path, [str])
    check_type('sub_d', sub_d, [bool])
    
    d_path = get_abs_path(d_path)
    if not os.path.isdir(d_path):
        raise NotADirectoryError(d_path)

    path_list = []
    if sub_d:
        for root, _, files in os.walk(d_path):
            for name in files:
                if f_extensions is None or name.lower().endswith(tuple(f_extensions)):
                    path_list.append(os.path.join(root, name))
    else:
        files = os.listdir(d_path)
        for name in files:
            if f_extensions is None or name.lower().endswith(tuple(f_extensions)):
                path_list.append(os.path.join(d_path, name))
    
    return np.array(path_list)

events_paths = get_path_list(events_dir)
ckpt_paths = get_path_list(ckpt_dir, f_extensions=['.ckpt'])

ckpts = []
for path in ckpt_paths:
    ckpt = torch.load(path)
    step = ckpt["auxiliary"]["current_step"]
    epoch = ckpt["auxiliary"]["current_epoch"]
    ckpts.append(f'{epoch}_{step}')
maps = defaultdict()

for path in events_paths:
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()  # carica tutti i dati


    mae_events = ea.Scalars("MAE_error")
    for e in mae_events:
        try:
            idx = ckpts.index(f'{e.step}_{e.wall_time}')
            maps[f'{e.step}_{e.wall_time}'] = [('MAE',e.value), idx]
        except ValueError:
            print("Elemento non presente nella lista")

    pr_events = ea.Scalars("pearsonr")
    for e in pr_events:
        maps[f'{e.step}_{e.wall_time}'].append(('pearsonr',e.value))
    
metric = "MAE"

# funzione che estrae il valore della metrica dalla lista di coppie
def get_metric_value(pairs, metric):
    return dict(pairs).get(metric, float("inf"))  # default inf se non trovato

# ordino le chiavi
sorted_keys = sorted(
    maps.keys(),
    key=lambda k: get_metric_value(maps[k], metric)
)

# prendo le prime 5
top5 = sorted_keys[:5]

# stampo i risultati
for k in top5:
    print(k, maps[k])

metric = 'pearsonr'
sorted_keys = sorted(
    maps.keys(),
    key=lambda k: get_metric_value(maps[k], metric)
)

# prendo le prime 5
top5 = sorted_keys[:5]

# stampo i risultati
for k in top5:
    print(k, maps[k])