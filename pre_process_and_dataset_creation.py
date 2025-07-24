import pre_process as pp
import utils as u
import dataset as dt
import numpy as np
import pandas as pd
import joblib as jb
from tqdm import tqdm
from collections import Counter
import time
import os
import argparse
import sys

parser = argparse.ArgumentParser(description="Script per il pre-processing e la creazione del dataset EEG.")
parser.add_argument('--input_edf_dir', type=str, required=True,
                    help='Directory contenente i file EEG originali in formato .edf.')
parser.add_argument('--output_raw_data_dir', type=str, default="/u/pmihelj/TESI-TRIENNALE/raw_data",
                    help='Directory di output per i dati EEG pre-processati (raw_data).')
parser.add_argument('--output_dataset_dir', type=str, default="/u/pmihelj/TESI-TRIENNALE/dataset",
                    help='Directory di output per i dataset finali (train/test).')
parser.add_argument('--log_file_path', type=str, default="/u/pmihelj/TESI-TRIENNALE/raw_data/log.csv",
                    help='Percorso del file di log per il pre-processing.')
parser.add_argument('--num_cpus', type=int, default=-1,
                    help='Numero di CPU da usare per la parallelizzazione. Usa -1 per tutti i core disponibili.')

args = parser.parse_args()

input_edf_dir = args.input_edf_dir
out_dir = args.output_raw_data_dir
log_file = args.log_file_path
out_directory = args.output_dataset_dir
n_jobs = args.num_cpus


print(f"DEBUG: Input EDF directory: {input_edf_dir}")
print(f"DEBUG: Output raw data directory: {out_dir}")
print(f"DEBUG: Output dataset directory: {out_directory}")
print(f"DEBUG: Log file path: {log_file}")
print(f"DEBUG: Number of CPUs to use: {n_jobs}")

if not os.path.isdir(input_edf_dir):
    print(f"ERRORE GRAVE: La directory di input EDF '{input_edf_dir}' non esiste!", file=sys.stderr)
    sys.exit(1)

#creo la lista dei path dei file contenenti gli eeg
path_list = u.get_path_list(input_edf_dir,['.edf'],True)
#print di controllo
print(type(path_list))
print(path_list.shape)
#creo i nomi dei file che salvero
path_id_list = [f"campione_{i}" for i in range(1, len(path_list) + 1)]
#creo il dataframe con path e nome
df_path = pd.DataFrame({'Path_id':path_id_list, 'Path':path_list})
#print di controllo
print(df_path.tail())
#salvo i path
md_file = "path.csv"
df_path.to_csv(md_file, index=False, encoding='utf-8-sig')
#creo la funzione da ripetere parallelamente
def worker(in_file, out_dir, out_prefix):
    try:
        dg = pp.data_gen(in_file, out_dir, out_prefix)
        dg.save_final_data(seg_len=5.0, merge_len=1)
    except:
        flag = False
        time.sleep(3.0)
    else:
        flag = True
    return in_file, flag

#se non esiste creo la directory di output
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

#carico i path e creo le task, cioè la coppia file-nome
tasks = [(Pt, Pt_id) for Pt, Pt_id in zip(df_path.Path, df_path.Path_id)]
print(f"DEBUG: Create {len(tasks)} task per la parallelizzazione.")

#definisco il numero di treadth in base ai core
if 'SLURM_CPUS_PER_TASK' in os.environ and args.num_cpus == -1:
    actual_n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])
elif args.num_cpus != -1:
    actual_n_jobs = args.num_cpus
else:
    actual_n_jobs = jb.cpu_count()

res = jb.Parallel(n_jobs=n_jobs, backend='loky')(jb.delayed(worker)(in_file, out_dir, out_f_id) for in_file, out_f_id in tqdm(tasks))
#salvo un file di log coi risultati dei work
df_log = pd.DataFrame(res, columns=['path', 'status'])
df_log.to_csv(log_file, index=False, encoding="utf-8-sig") 
print(f"DEBUG: File di log '{log_file}' salvato.")

#semplice controllo delle dimensioni dei file
clips = []
chans = []
freqs = []
for path in tqdm(path_list):
    file = np.load(path)
    clips.append(file.shape[0])
    chans.append(file.shape[1])
    freqs.append(file.shape[2])
print("number of clips: ", Counter(clips))
print("number of channels: ", Counter(chans))
print("different frequencies: ", Counter(freqs))
print("DEBUG: Controllo dimensioni file completato.")

files_directory = out_dir
dt.make_save_dataset(f_dir=files_directory, out_dir=out_directory)
print("DEBUG: Creazione dataset completata.")