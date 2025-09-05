"""
from utils import get_path_list, get_raw
from collections import Counter
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Dati di esempio
modelli = ['Mod1', 'Mod2', 'Mod3', 'Mod4', 'Mod5']
bande = ['Banda1', 'Banda2', 'Banda3', 'Banda4', 'Banda5']

# valori[modello][banda] -> trasponiamo per avere valori[banda][modello]
valori = [
    [5, 3, 4, 2, 1],
    [2, 4, 5, 3, 2],
    [3, 5, 2, 4, 3],
    [4, 2, 3, 5, 4],
    [1, 3, 4, 2, 5]
]

# Trasponiamo per avere liste per bande
valori_per_banda = np.array(valori).T  # ora valori_per_banda[banda][modello]

# Configurazione dell'istogramma
x = np.arange(len(bande))
width = 0.15

for i in range(len(modelli)):
    plt.bar(x + i*width, valori_per_banda[:, i], width, label=modelli[i])

plt.xticks(x + 2*width, bande)  # centratura delle bande
plt.ylabel("Valore")
plt.xlabel("Bande")
plt.title("Istogramma raggruppato per bande e modelli")
plt.legend()
plt.tight_layout()
plt.show()
"""
"""
path_list = get_path_list("D:/nchsdb/sleep_data", ['.edf'])
print(len(path_list))

lenght_h = []
fs = []
cha = []

for path in tqdm(path_list):
    raw = get_raw(path, preload=True)
    fs.append(raw.info['sfreq'])
    cha.append(raw.info['ch_names'])
    eeg = raw.get_data()
    lenght_h.append(eeg.shape[1]/int(fs[0])/60/60)
    break

freq = Counter(fs)
chs = Counter(np.array(cha).flatten())
lenght_h = np.array(lenght_h)
max_len = np.max(lenght_h)
min_len = np.min(lenght_h)
mean_len = np.mean(lenght_h)

print('counter delle frequenze',freq)
print('##################################################################################################')
print(f'Lunghezza media: {mean_len:.2f}   Lunghezza massima: {max_len:.2f}  Lunghezza minima: {min_len:.2f}')
print('###################################################################################################')
print('Counter dei canali: ', chs)
"""

"""
import mne
import numpy as np
eeg = mne.io.read_raw_edf("D:/nmt_scalp_eeg_dataset/normal/eval/0000024.edf")
print(len(eeg.info['ch_names']))
print(eeg.get_data().shape[1]/250/60)
"""

"""
import matplotlib.pyplot as plt
import mne
import numpy as np

# Lista canali
ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 
            'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 
            'T3', 'T4', 'T5', 'T6', 'O1', 'O2']

n_channels = len(ch_names)

# Valori casuali per i barplot (esempio: 5 dataset)
np.random.seed(42)
values_list = [np.random.rand(n_channels) for _ in range(5)]
models = [f"Model {i+1}" for i in range(n_channels)]
y_lab = [f"Y {i+1}" for i in range(5)]
titles = [f"Barplot {i+1}" for i in range(5)]

# Valori per la topomap (un array di dimensione n_channels)
topo_values = np.random.randn(n_channels)

# Creo l'info MNE
info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

# Creo Evoked fittizio per la topomap
evoked = mne.EvokedArray(topo_values[:, np.newaxis], info)

# Creo figure con 6 subplot (5 barplot + 1 topomap)
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs = axs.flatten()

x = np.arange(len(models))

# 5 grafici a barre
for j, (vals, ax) in enumerate(zip(values_list, axs[:-1])):  # lascia ultimo per topomap
    ax.bar(x, vals, color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90)
    ax.set_ylabel(y_lab[j])
    ax.set_title(titles[j])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

# Ultimo subplot: topomap
mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=axs[-1], show=False,
                     cmap="RdBu_r", contours=0)
axs[-1].set_title("Topomap")

plt.tight_layout()
plt.show()
"""
"""
import numpy as np
import mne
import matplotlib.pyplot as plt

# Lista canali
ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 
            'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 
            'T3', 'T4', 'T5', 'T6', 'O1', 'O2']

n_channels = len(ch_names)

# Valori casuali per la topomap
values = np.random.randn(n_channels)

# Creo info e montage
info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

# Creo Evoked fittizio
evoked = mne.EvokedArray(values[:, np.newaxis], info)

# Creo figure
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Topomap senza contorni
mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=axs[0], show=False,
                     cmap="RdBu_r", contours=0)
axs[0].set_title("contours=0 (nessuna linea)")

# Topomap con 6 contorni
mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=axs[1], show=False,
                     cmap="RdBu_r", contours=6)
axs[1].set_title("contours=6 (default)")

plt.tight_layout()
plt.show()
"""
"""
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib

# Lista canali
ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8',
            'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4',
            'T3', 'T4', 'T5', 'T6', 'O1', 'O2']

n_channels = len(ch_names)

# Valori casuali per la topomap
values = np.random.randn(n_channels)

# Creo info e assegno il montage standard
info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

# Creo Evoked fittizio
evoked = mne.EvokedArray(values[:, np.newaxis], info)

# Creo figura
fig, ax = plt.subplots(figsize=(6,6))

# Topomap con puntini grandi e colore solo all’interno della testa
im, cn = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info,
                              axes=ax,
                              show=False,
                              cmap='jet',
                              contours=6,
                              sensors=True)

# Individuo solo i punti degli elettrodi e aumento la dimensione
for coll in ax.collections:
    if isinstance(coll, matplotlib.collections.PathCollection):  # solo scatter
        coll.set_sizes([10])  # aumenta la dimensione dei marker
        coll.set_facecolor('k')            # colore pieno (nero)
        coll.set_edgecolor('k')            # bordo nero
        coll.set_alpha(1.0) 


ax.set_title("Topomap EEG con puntini grandi", fontsize=14)
fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.05, label='Ampiezza[uV]', format="%.1f")
plt.show()
"""

"""
from utils import get_path_list
import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt

path = get_path_list("D:/nmt_scalp_eeg_dataset", f_extensions=['.edf'], sub_d=True)
print(len(path))

age_df = pd.read_csv("D:/nmt_scalp_eeg_dataset/Labels.csv")
ages = []
for p in path:
    code = os.path.basename(p)
    ages.append(age_df.loc[age_df['recordname'] == code, 'age'].iloc[0])
ages = dict(Counter(ages))
ages = dict(sorted(ages.items()))  


df = pd.DataFrame.from_dict(ages, orient='index', columns=['count'])
df.plot(kind='bar', legend=False)

plt.xlabel("Età")
plt.ylabel("Frequenza")
plt.show()
"""

"""
from deploy import get_orig_rec_latent, DeployBaseline
from utils import check_channel_names, get_raw
import mne
eeg = get_raw("D:/nmt_scalp_eeg_dataset/abnormal/eval/0000036.edf")
check_channel_names(raw_obj=eeg, verbose=False)
eeg = eeg.get_data()
model = DeployBaseline("C:/Users/Pietro/Desktop/TESI/TESI-TRIENNALE/models/PCA_whole.pkl")
_,_,lat = get_orig_rec_latent(eeg, model)

print(lat.shape)
"""

"""
from utils import get_path_list
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

paths = get_path_list("D:/nmt_scalp_eeg_dataset", ['.edf'], True)
for j,path in enumerate(paths):
    paths[j] = os.path.basename(path)
df = pd.read_csv("D:/nmt_scalp_eeg_dataset/Labels.csv")

df = df.copy()
df['age'] = df['age'].astype(int)
df['gender'] = df['gender'].str.lower()

# Indice completo di età (uno per anno)
min_age = df['age'].min()
max_age = df['age'].max()
age_idx = pd.Index(np.arange(min_age, max_age + 1), name='age')

# Conteggi per età e genere, riallineati all’indice completo
counts = (df.groupby(['age', 'gender']).size()
            .unstack('gender')
            .reindex(age_idx, fill_value=0))

m = counts.get('male', pd.Series(0, index=age_idx)).to_numpy()
f = counts.get('female', pd.Series(0, index=age_idx)).to_numpy()

y = age_idx.values  # posizioni Y = età

fig, ax = plt.subplots(figsize=(8, max(4, len(age_idx)/6)))

# Barre: maschi a sinistra (valori negativi), femmine a destra
ax.barh(y, -m, align='center', label='Male')
ax.barh(y,  f, align='center', label='Female')

# Linea dello zero
ax.axvline(0, linewidth=1)

# Tick ogni 10 anni, allineati alle stesse posizioni Y
ticks = np.arange((min_age//10)*10, (max_age//10 + 1)*10 + 1, 10)
ax.set_yticks(ticks)
ax.set_yticklabels([str(t) for t in ticks])

# Limiti Y centrati sulle barre, così i tick cadono al centro
ax.set_ylim(min_age - 0.5, max_age + 0.5)

ax.set_xlabel('Count')
ax.set_ylabel('Age')
ax.legend()
ax.grid(axis='x', linestyle=':', alpha=0.5)
# ax.invert_yaxis()  # opzionale, per avere età crescenti verso il basso
plt.tight_layout()
plt.show()
"""

"""
from utils import get_raw, check_channel_names
from deploy import get_orig_rec_latent, load_models
import numpy as np

eeg = get_raw("D:/nmt_scalp_eeg_dataset/normal/eval/0000024.edf")
eeg1 = get_raw("D:/nmt_scalp_eeg_dataset/normal/eval/0000044.edf")
check_channel_names(raw_obj=eeg, verbose=False)
check_channel_names(raw_obj=eeg1, verbose=False)
eeg = eeg.get_data()
eeg1 = eeg1.get_data()
model = load_models('PCA', "C:/Users/Pietro/Desktop/TESI/TESI-TRIENNALE/models/PCA_whole.pkl", [1])
_,_,latent = get_orig_rec_latent(eeg, model)
_,_,latent1 = get_orig_rec_latent(eeg1, model)
print(latent1.shape)
latents = np.concat([np.hstack([latent.reshape(-1,50), np.full((latent.reshape(-1,50).shape[0],1),0)]), np.hstack([latent1.reshape(-1,50), np.full((latent1.reshape(-1,50).shape[0],1),1)])], axis=0)
print(latents.shape)
"""

import numpy as np
 
c = np.zeros((3,5))
c = np.hstack([c, np.full((c.shape[0],1),1)])
print(c)