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