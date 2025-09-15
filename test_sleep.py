import pandas as pd
import mne
import numpy as np
from collections import defaultdict
from utils import select_bipolar, to_monopolar, strip_eeg_prefix

# Carica il TSV
df = pd.read_csv("test_sleep/10000_17728.tsv", sep='\t')

# Definisci i sleep stages che ti interessano
sleep_stages = {"Sleep stage W", "Sleep stage N1", "Sleep stage N2", "Sleep stage N3"}

# Lista dei segmenti (start, end, stage)
segments = []

for _, row in df.iterrows():
    desc = row['description']
    if desc in sleep_stages:
        start = row['onset']
        end = start + row['duration']  # duration è in secondi
        segments.append((start, end, desc))

# Ordina per inizio, opzionale
segments.sort(key=lambda x: x[0])
segments.insert(0,(0, segments[0][0], 'Wakefullness'))
print(len(segments))
# Mostra i primi segmenti
for seg in segments[:10]:
    print(seg)

raw = mne.io.read_raw_edf('test_sleep/10000_17728.edf')
strip_eeg_prefix(raw)
raw, _ = select_bipolar(raw)
raw_mono, max_diff = to_monopolar(raw)
print(raw.info['ch_names'], max_diff)

intervalls = defaultdict(list)

eeg = raw.get_data()
print(eeg.max())
sf = raw.info['sfreq']
raw.close()

for (start, end, stage) in segments:
    intervalls[stage].append(eeg[:,int(start*sf):int(end*sf)])

for stage in intervalls:
    intervalls[stage] = np.array(intervalls[stage])

for stage in intervalls:
    print(intervalls[stage].shape)
