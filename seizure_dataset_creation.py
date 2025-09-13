"""
#monopolarizzazione dei canali per il seizure dataset

from utils import get_path_list, to_monopolar, select_bipolar, get_raw, check_channel_names
import warnings
import numpy as np
from collections import Counter
from tqdm import tqdm
import mne
import os
import gc
warnings.filterwarnings("ignore")

channels = ['FP1', 'F7', 'O1', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'FZ', 'CZ', 'PZ', 'T7', 'P7', 'T8', 'P8']
rename_dict = {'T7':'T3', 'P7':'T5', 'T8':'T4', 'P8':'T6'}

raws = get_path_list("D:/CHB-MIT_seizure/chb-mit-scalp-eeg-database-1.0.0", f_extensions=['.edf'], sub_d=True)
print('Numero di campioni: ', len(raws))

i=0
j=0
erased = []

for raw_path in tqdm(raws):
    raw = get_raw(raw_path)
    raw, flag = select_bipolar(raw)
    if not flag:
        save_path = raw_path.replace("CHB-MIT_seizure", "seizure_monopolar_dataset")
        if any(ch not in raw.info['ch_names'] for ch in channels):
            j=j+1
            erased.append(raw)
            continue
        raw.pick(channels)
        raw.rename_channels(rename_dict)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mne.export.export_raw(save_path, raw, fmt='edf', physical_range=(-1.5*0.005, 1.5*0.005), overwrite=True, verbose=False)
        i=i+1
        del raw
        gc.collect()
        continue
    mono,_ = to_monopolar(raw, ref=['CS', 'CZ']) 
    if any(ch not in mono.info['ch_names'] for ch in channels):
        j=j+1
        erased.append(raw_path)
        continue
    mono.pick(channels)
    mono.rename_channels(rename_dict)  
    save_path = raw_path.replace("CHB-MIT_seizure", "seizure_monopolar_dataset")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #mne.export.export_raw(save_path, mono, fmt='edf', physical_range=(-1.5*0.005, 1.5*0.005), overwrite=True, verbose=False)
    i = i+1
    for var_name in ['raw', 'mono', 'data']:
        if var_name in globals():
            del globals()[var_name]
    gc.collect()

print('File salvati: ',i)
print('File eliminati per canali mancanti: ',j)
"""


import pandas as pd
import re
from mne import concatenate_raws
import os
from tqdm import tqdm
import warnings
from utils import get_path_list, get_raw, check_channel_names
from deploy import load_models, get_orig_rec_latent
import numpy as np
import gc
warnings.filterwarnings("ignore")

save_dir = 'seizure_datset'

seizure_info_paths = get_path_list("D:/CHB-MIT_seizure", f_extensions=['.txt'], sub_d=True)
def parse_summary_txt(txt_path):
    data = []
    current = {}
    sampling_rate = None
    channels = []

    seizure_start_pattern = re.compile(r'Seizure(?: \d+)? Start Time: (\d+) seconds')
    seizure_end_pattern = re.compile(r'Seizure(?: \d+)? End Time: (\d+) seconds')

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Frequenza di campionamento
            if line.startswith("Data Sampling Rate:"):
                sampling_rate = float(line.split(":",1)[1].strip().split()[0])

            # Lista canali
            elif line.startswith("Channels in EDF Files:"):
                channels = []
                continue

            elif line.startswith("Channel "):
                label = line.split(":",1)[1].strip()
                channels.append(label)

            # Inizio di un nuovo file
            elif line.startswith("File Name:"):
                if 'File Name' in current:
                    current.setdefault("Seizure Start Times", [])
                    current.setdefault("Seizure End Times", [])
                    if current.get("Seizure Count", 0) > 0 and not current["Seizure Start Times"]:
                        print(f"Warning: {current['File Name']} ha Seizure Count > 0 ma nessun inizio/fine segnalato")
                    data.append(current.copy())
                current = {"File Name": line.split(":",1)[1].strip()}

            elif line.startswith("File Start Time:"):
                current["Start Time"] = line.split(":",1)[1].strip()

            elif line.startswith("File End Time:"):
                current["End Time"] = line.split(":",1)[1].strip()

            elif line.startswith("Number of Seizures in File:"):
                current["Seizure Count"] = int(line.split(":",1)[1].strip())

            else:
                # Cattura tutti i Seizure Start/End numerati o non numerati
                m_start = seizure_start_pattern.match(line)
                m_end = seizure_end_pattern.match(line)
                if m_start:
                    current.setdefault("Seizure Start Times", []).append(int(m_start.group(1)))
                elif m_end:
                    current.setdefault("Seizure End Times", []).append(int(m_end.group(1)))

        # aggiungo l'ultimo file
        if current:
            current.setdefault("Seizure Start Times", [])
            current.setdefault("Seizure End Times", [])
            if current.get("Seizure Count",0) > 0 and not current["Seizure Start Times"]:
                print(f"Warning: {current['File Name']} ha Seizure Count > 0 ma nessun inizio/fine segnalato")
            data.append(current)

    # creo il DataFrame
    df = pd.DataFrame(data)

    # metadati generali
    metadata = {
        "Sampling Rate": sampling_rate,
        "Channels": channels
    }

    return metadata, df

def time_to_seconds(t):
    if pd.isna(t):
        return None  # gestisce valori mancanti
    h, m, s = map(int, str(t).split(":"))
    return h*3600 + m*60 + s

def duration_seconds(row):
    start_sec = time_to_seconds(row["Start Time"])
    end_sec = time_to_seconds(row["End Time"])
    
    if start_sec is None or end_sec is None:
        return None
    
    # se fine < inizio significa giorno successivo
    if end_sec < start_sec:
        end_sec += 24*3600
    
    return end_sec - start_sec

def seizure_duration(row):
    starts = row["Seizure Start Times"]
    ends = row["Seizure End Times"]
    # sottrazione elemento per elemento e somma delle durate
    durations = [end - start for start, end in zip(starts, ends)]
    return sum(durations)

remove = ['chb12_27.edf','chb12_28.edf','chb12_29.edf']

dfs = []
for path in seizure_info_paths:  
    metadata, df_files = parse_summary_txt(path)
    dfs.append(df_files)

total_df = pd.concat(dfs, ignore_index=True)
total_df = total_df[~total_df['File Name'].isin(remove)]
seizures_starts = [s if s else [None] for s in total_df['Seizure Start Times']]
seizures_ends = [s if s else [None] for s in total_df['Seizure End Times']]

raw_paths = get_path_list("D:/seizure_monopolar_dataset", f_extensions=['.edf'], sub_d=True)
for raw_path, start, end in tqdm(zip(raw_paths, seizures_starts, seizures_ends)):
    raw = get_raw(raw_path)
    if start[0] is None and end[0] is None:
        save_clean = raw_path.replace("seizure_monopolar_dataset", "seizure_dataset/normal")
        os.makedirs(os.path.dirname(save_clean), exist_ok=True)
        raw.export(save_clean, fmt='edf', overwrite=True, verbose=False)
        continue
    seizure_raws = []
    for st, e in zip(start,end):
        try:
            seizure_raws.append(raw.copy().crop(tmin=st, tmax=e))
        except ValueError:
            print(f'File: {raw_path} ha un end seizure maggiore della lunghezza del file')
    
    keep_segments = []
    if start[0] > 0:
        try:
            keep_segments.append(raw.copy().crop(tmin=0, tmax=start[0]))
        except ValueError:
            print(f'File: {raw_path} ha un tmax troppo grande prima della prima seizure')
            keep_segments.append(raw.copy().crop(tmin=0, tmax=raw.times[-1]))
            continue
    for i in range(min(len(start) - 1, len(end) - 1)):
        if end[i] is None or start[i] is None:
            print(f'Uno tra end e start del file. {raw_path} è None')
            continue
        try:
            keep_segments.append(raw.copy().crop(tmin=end[i], tmax=start[i+1]))
        except ValueError:
            print(f'File: {raw_path} ha un tmax troppo grande tra le seizure')
    if end[-1] is None:
        print(f'File: {raw_path} ha None come ultima seizure')
        continue
    if end[-1] < raw.times[-1]:
        try:
            keep_segments.append(raw.copy().crop(tmin=end[-1], tmax=raw.times[-1]))
        except ValueError:
            print(f'File: {raw_path} ha un tmax troppo grande dopo la ultima seizure')
    raw_clean = concatenate_raws(keep_segments)
    save_clean = raw_path.replace("seizure_monopolar_dataset", "seizure_dataset/normal")
    save_seizure = raw_path.replace("seizure_monopolar_dataset", "seizure_dataset/seizure")
    os.makedirs(os.path.dirname(save_clean), exist_ok=True)
    os.makedirs(os.path.dirname(save_seizure), exist_ok=True)

    raw_clean.export(save_clean, fmt='edf', overwrite=True, verbose=False)
    for i,sez in enumerate(seizure_raws):
        save_seizure = save_seizure.replace(os.path.splitext(os.path.basename(save_seizure))[0], os.path.splitext(os.path.basename(save_seizure))[0]+f'_{i}')
        sez.export(save_seizure, fmt='edf', overwrite=True, verbose=False)


save_norm = "D:/DS_seiz/normal/"
save_seiz = "D:/DS_seiz/abnormal/"
os.makedirs(save_norm, exist_ok=True)
os.makedirs(save_seiz, exist_ok=True)
normal_paths = get_path_list("D:/seizure_dataset/normal", f_extensions=['.edf'], sub_d=True)
print('Caricamento modelli')
model = load_models(model='VAEEG', save_files=['models/VAEEG/delta_band.ckpt', 'models/VAEEG/theta_band.ckpt', 'models/VAEEG/alpha_band.ckpt', 'models/VAEEG/low_beta_band.ckpt', 'models/VAEEG/high_beta_band.ckpt'], params=[[8], [10], [12], [10], [10]])
print('Modelli caricati')
print('###########################################################################')
print('Inizio salvataggio normali:')
for i,norm in enumerate(normal_paths):
    raw_norm = get_raw(norm)
    check_channel_names(raw_norm, verbose=False)
    eeg_norm = raw_norm.get_data().astype(np.float32)
    raw_norm.close()
    del raw_norm
    pi, pj, latent_norm = get_orig_rec_latent(raw=eeg_norm, model=model, fs=256)
    base_name = os.path.splitext(os.path.basename(norm))[0]
    np.save(os.path.join(save_norm, base_name + '.npy'), latent_norm)
    print(f'Salvato file {norm}')
    del eeg_norm, latent_norm, pi, pj
    gc.collect()
print('Fine salvataggio normali')
print('############################################################################')

print('Inizio salvataggio seizure')
seizure_paths = get_path_list("D:/seizure_dataset/normal", f_extensions=['.edf'], sub_d=True)
for j,seiz in enumerate(seizure_paths):
    raw_seiz = get_raw(seiz)
    check_channel_names(raw_seiz, verbose=False)
    eeg_seiz = raw_seiz.get_data().astype(np.float32)
    raw_seiz.close()
    del raw_seiz
    ti, tj, latent_seiz = get_orig_rec_latent(raw=eeg_seiz, model=model, fs=256)
    base_name = os.path.splitext(os.path.basename(seiz))[0]
    np.save(os.path.join(save_seiz, base_name + '.npy'), latent_seiz)
    print(f'Salvato file {seiz}')
    del eeg_seiz, latent_seiz, ti, tj
    gc.collect()
print('Fine salvataggio seizure')
print('############################################################################')
print(f'File normali salvati: {i}, File seizure salvati: {j}')
print('Done')










