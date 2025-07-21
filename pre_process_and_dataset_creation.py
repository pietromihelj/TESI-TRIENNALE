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

#creo la lista dei path dei file contenenti gli eeg
path_list = u.get_path_list('D:/isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf',['.edf'],True)
#print di controllo
print(type(path_list))
print(path_list.shape)
#creo i nomi dei file che salvero
path_id_list= [f"campione_{i}" for i in range(1, 2994)]
#creo il dataframe con path e nome
df_path = pd.DataFrame({'Path_id':path_id_list, 'Path':path_list})
#print di controllo
print(df_path.tail())
#salvo i path
df_path.to_csv('path.csv', index=False, encoding='utf-8-sig')
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

#definisco i path delle variabili
md_file = "path.csv"
out_dir = "D:/raw_data5s"
log_file = "D:/raw_data5s/log.csv"

#se non esiste creo la directory di output
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

#carico i path e creo le task, cioè la coppia file-nome
df_path = pd.read_csv(md_file)
tasks = [(Pt, Pt_id) for Pt, Pt_id in zip(df_path.Path, df_path.Path_id)]

#definisco il numero di treadth in base ai core
n_jobs = jb.cpu_count()
res = jb.Parallel(n_jobs=n_jobs, backend='loky')(jb.delayed(worker)(in_file, out_dir, out_f_id) for in_file, out_f_id in tqdm(tasks))
#salvo un file di log coi risultati dei work
df_log = pd.DataFrame(res, columns=['path', 'status'])
df_log.to_csv(log_file, index=False, encoding="utf-8-sig") 

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

#creo le directory contenenti i datasets di train e test
files_directory = "D:/raw_data5s"
out_directory = "D:/Dataset5s"
dt.make_save_dataset(f_dir=files_directory, out_dir=out_directory)