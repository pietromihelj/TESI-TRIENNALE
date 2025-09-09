"""
from utils import get_path_list
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

Grafico piramide delle età per il dataset della age regression

df = pd.read_csv("D:/nmt_scalp_age_dataset/Labels.csv")
paths = get_path_list('D:/nmt_scalp_age_dataset', f_extensions=['.edf'], sub_d=True)

file_names = []
for path in paths:
    file_names.append(os.path.basename(path))

df = df[df["recordname"].isin(file_names)]
df = df[df["gender"].isin(["male","female"])]
print('Numero di eeg: ', len(file_names))


bins = np.arange(0, 91, 1)
df["age_group"] = pd.cut(df["age"], bins=bins, right=False)

# conteggio per fascia e genere
counts = df.groupby(["age_group", "gender", "label"]).size().unstack(fill_value=0)
counts.columns = counts.columns.str.lower()
counts_reset = counts.reset_index()

# Separiamo maschi e femmine
male = counts_reset[counts_reset['gender'] == 'male']
female = counts_reset[counts_reset['gender'] == 'female']

# Creiamo etichette combinate
labels = male['age_group'].astype(str)  # o female['age_group'], sono uguali

fig, ax = plt.subplots(figsize=(10, 8))

# Barre divergenti (negativi per male)
ax.barh(labels, -male['normal'], color="#7f80e6", alpha=0.8, label='Male normal (24.6%, N=85)')
ax.barh(labels, -male['abnormal'], color="#99d9d4",alpha = 0.8, label='Male abnormal (75.4%, N=260)')
ax.barh(labels, female['normal'], color="#bc8485", alpha=0.8, label='Female normal (20.5%, N=40)')
ax.barh(labels, female['abnormal'], color='#f7beb8', alpha=0.8, label='Female abnormal (79.5%, N=155)')


ax.grid(True, axis='both', linestyle='--', linewidth=0.7, color='lightgray', alpha=0.8)

ax.set_yticks(np.arange(0,91,10))
ax.set_yticklabels(['0','10', '20', '30', '40', '50','60', '70', '80','90'])
ax.set_xticks(np.arange(-20,21,5))
ax.set_xticklabels(['20','15','10','5','0','5','10','15','20'])
ax.set_xlabel('Count')
ax.set_ylabel('Age (years)')
ax.set_title('male (63.9%, N=345)              female(36.1%, N=195)')
ax.legend(loc='upper left', frameon=False)
ax.set_ylim(0, 90)
plt.show()

ages = df['age']
print(np.mean(ages))
print(np.sqrt(np.var(ages)))
"""



"""
Funzione per la monopolarizzazione dei canali

import numpy as np
import mne

def bipolar_to_monopolar(data_bipolar, bipolar_ch_names, sfreq=256):

    # Troviamo tutti i nomi di elettrodi presenti nei bipolari
    electrodes = set()
    for ch in bipolar_ch_names:
        electrodes.update(ch.split('-'))
    electrodes = sorted(list(electrodes))

    n_samples = data_bipolar.shape[1]
    n_elec = len(electrodes)

    # Costruisco matrice A (n_bipolar x n_monopolar) e vettore y (bipolar)
    A = np.zeros((len(bipolar_ch_names), n_elec))
    for i, ch in enumerate(bipolar_ch_names):
        e1, e2 = ch.split('-')
        idx1 = electrodes.index(e1)
        idx2 = electrodes.index(e2)
        A[i, idx1] = 1
        A[i, idx2] = -1

    # Risolvo il sistema lineare con minimo quadrati
    X_monopolar = np.linalg.lstsq(A, data_bipolar, rcond=None)[0]

    # Creo info MNE
    info = mne.create_info(ch_names=electrodes, sfreq=sfreq, ch_types='eeg')

    # Creo RawArray
    raw = mne.io.RawArray(X_monopolar, info)

    # Applico riferimento medio
    raw.set_eeg_reference('average', projection=False)
    
    return raw

# Esempio di utilizzo
bipolar_ch_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
                     'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                     'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                     'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                     'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9',
                     'FT9-FT10', 'FT10-T8', 'T8-P8']

data_bipolar = np.random.randn(len(bipolar_ch_names), 1000)  # esempio

raw_monopolar = bipolar_to_monopolar(data_bipolar, bipolar_ch_names, sfreq=256)

print(raw_monopolar.info)
raw_monopolar.plot(n_channels=10, scalings='auto')

"""

"""
#risultati dello studio del dataset pèer le seizure

paths = get_path_list("D:/CHB-MIT_seizure/chb-mit-scalp-eeg-database-1.0.0", f_extensions=['.edf'], sub_d=True)
print('Numero di eeg: ', len(paths))

subdirs = [entry.name for entry in os.scandir("D:/CHB-MIT_seizure/chb-mit-scalp-eeg-database-1.0.0") if entry.is_dir()]
print('Numero di pazienti: ', len(subdirs))

paths = get_path_list("D:/CHB-MIT_seizure/chb-mit-scalp-eeg-database-1.0.0", f_extensions=['.txt'], sub_d=True)



import pandas as pd
from datetime import datetime, timedelta
import re

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


# Utilizzo:
print(len(paths))
dfs = []
for path in paths:  
    metadata, df_files = parse_summary_txt(path)
    dfs.append(df_files)

total_df = pd.concat(dfs, ignore_index=True)

def time_to_seconds(t):
    if pd.isna(t):
        return None  # gestisce valori mancanti
    h, m, s = map(int, str(t).split(":"))
    return h*3600 + m*60 + s

# funzione per calcolare la durata in secondi
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

total_df["Duration (s)"] = total_df.apply(duration_seconds, axis=1)
total_df["Seizure Duration (s)"] = total_df.apply( lambda row: sum(end - start for start, end in zip(row["Seizure Start Times"], row["Seizure End Times"])), axis=1)
print(total_df.head())

print('Durata totale degli edf: ',np.sum(total_df['Duration (s)'])/60/60)
print('Durata totale delle seizure: ', np.sum(total_df['Seizure Duration (s)']/60/60))
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

for raw_path in tqdm(raws):
    raw = get_raw(raw_path)
    raw, flag = select_bipolar(raw)
    if not flag:
        save_path = raw_path.replace("CHB-MIT_seizure", "seizure_monopolar_dataset")
        if any(ch not in raw.info['ch_names'] for ch in channels):
            j=j+1
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
        continue
    mono.pick(channels)
    mono.rename_channels(rename_dict)  
    save_path = raw_path.replace("CHB-MIT_seizure", "seizure_monopolar_dataset")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mne.export.export_raw(save_path, mono, fmt='edf', physical_range=(-1.5*0.005, 1.5*0.005), overwrite=True, verbose=False)
    i = i+1
    for var_name in ['raw', 'mono', 'data']:
        if var_name in globals():
            del globals()[var_name]
    gc.collect()

print('File salvati: ',i)
print('File eliminati per canali mancanti: ',j)

eeg = mne.io.read_raw_edf("D:/seizure_monopolar_dataset/chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_01.edf")
