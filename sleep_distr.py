"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import get_path_list
from sklearn.preprocessing import StandardScaler

# --- Caricamento dati ---
eeg_W = get_path_list("D:/sleep/VAEEG/W", f_extensions=['.npy'], sub_d=True)
eeg_R = get_path_list("D:/sleep/VAEEG/R", f_extensions=['.npy'], sub_d=True)
eeg_N3 = get_path_list("D:/sleep/VAEEG/N3", f_extensions=['.npy'], sub_d=True)
eeg_N2 = get_path_list("D:/sleep/VAEEG/N2", f_extensions=['.npy'], sub_d=True)
eeg_N1 = get_path_list("D:/sleep/VAEEG/N1", f_extensions=['.npy'], sub_d=True)

data = [eeg_N1, eeg_N2, eeg_N3, eeg_R, eeg_W]

# --- Pulizia clip corrotte ---
inf_clip = 0
inf_num_clip = []
data_mean = []
for stage in data:
    stage_mean = []
    for eeg in stage:
        eeg_data = np.load(str(eeg), allow_pickle=True)
        finite_vals = eeg_data[np.isfinite(eeg_data)]
        max_val = np.max(finite_vals)
        min_val = np.min(finite_vals)
        # sostituisco -inf con min e +inf con max
        eeg_data[eeg_data == np.inf] = max_val
        eeg_data[eeg_data == -np.inf] = min_val
        stage_mean.append(np.mean(eeg_data, axis=0))
        del eeg_data
    data_mean.append(np.concatenate(stage_mean))



# --- Suddivisione bande ---
deltas, thetas, alphas, low_betas, high_betas = [], [], [], [], []
for stage_mean in data_mean:
    deltas.append(stage_mean[:, :8])        # delta = prime 8 colonne
    thetas.append(stage_mean[:, 8:18])      # theta = colonne 8-17
    alphas.append(stage_mean[:, 18:30])     # alpha = colonne 18-29
    low_betas.append(stage_mean[:, 30:40])  # low beta = colonne 30-39
    high_betas.append(stage_mean[:, 40:])   # high beta = colonne 40 in poi

bands = {
    "delta": deltas,
    "theta": thetas,
    "alpha": alphas,
    "low_beta": low_betas,
    "high_beta": high_betas,
}

# --- PCA ---
pca = PCA(2)
scaler = StandardScaler()
bands_pca = {band: [] for band in bands}

for band, stages_list in bands.items():
    # Concateno tutti gli stage della banda
    all_stage_data = np.concatenate(stages_list, axis=0)
    all_stage_data = all_stage_data[~np.isnan(all_stage_data).any(axis=1)]
    
    # Fit scaler e PCA sull'intera banda
    scaler = StandardScaler()
    all_stage_scaled = scaler.fit_transform(all_stage_data)
    
    pca = PCA(2)
    pca.fit(all_stage_scaled)
    
    # Trasformo ogni stage separatamente
    bands_pca[band] = []
    for stage_clips in stages_list:
        stage_clips_clean = stage_clips[~np.isnan(stage_clips).any(axis=1)]
        stage_scaled = scaler.transform(stage_clips_clean)
        stage_pca = pca.transform(stage_scaled)
        bands_pca[band].append(stage_pca)

for band, stage_list in bands_pca.items():
    print(band)
    for stage in stage_list:
        print(f'Forma stage: {stage.shape}')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sleep_stages = ["N1", "N2", "N3", "R", "W"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

bands_list = list(bands_pca.keys())
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), sharey=False)

sample_size = 10000  # massimo numero di punti da usare per KDE

for col, band in enumerate(bands_list):
    stage_list = bands_pca[band]
    
    for row, dim in enumerate([0, 1]):  # row 0 -> PC1, row 1 -> PC2
        ax = axes[row, col]
        
        for i, stage in enumerate(stage_list):
            values = stage[:, dim]
            
            # Campiona se ci sono troppi punti
            if len(values) > sample_size:
                values = np.random.choice(values, sample_size, replace=False)
            
            # Imposta bandwidth diversa per N1 per lisciare il picco
            if sleep_stages[i] == "N1":
                bw = 0.6  # più liscia per N1
            else:
                bw = 0.3  # standard per gli altri stage
            
            # KDE
            kde = gaussian_kde(values, bw_method=bw)
            x_grid = np.linspace(np.min(values), np.max(values), 500)
            kde_values = kde(x_grid)
            
            # Normalizzazione KDE: integrale = 1
            kde_values /= kde_values.sum() * (x_grid[1] - x_grid[0])
            
            # Plot con trasparenza per sovrapposizione
            ax.plot(x_grid, kde_values, label=sleep_stages[i], color=colors[i], alpha=0.6)
        
        # Asse x con 3 tick equidistanti tra min e max
        x_ticks = np.linspace(np.min(values), np.max(values), 3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{t:.2f}' for t in x_ticks])
        
        # Tick y automatici
        ax.tick_params(axis='y', labelsize=10)
        
        ax.set_xlabel(f'Componente {dim+1}', fontsize=10)
        ax.set_ylabel('Densità', fontsize=10)
        
        # Titolo sopra la prima riga con il nome della banda
        if row == 0:
            ax.set_title(f'{band}', fontsize=18, fontweight='bold')
        
        # Legenda per ogni subplot
        ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig("sleep_results/sleep_stage_distr.png", dpi=300)
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt

# Caricamento dati
f_c = np.load("sleep_results/FastICAconf_matr.npy")
mat_norm = f_c / f_c.sum(axis=1, keepdims=True)  # valori tra 0 e 1

# Nomi delle classi
class_names = ['N1', 'N2', 'N3', 'R', 'W']

# Visualizzazione
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(mat_norm, cmap='Blues')

# Annotazioni (valori tra 0 e 1, 1 decimale)
for i in range(mat_norm.shape[0]):
    for j in range(mat_norm.shape[1]):
        ax.text(j, i, f'{mat_norm[i,j]:.1f}', ha='center', va='center', color='black')

# Etichette assi
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

# Griglia
ax.set_xticks(np.arange(-.5, len(class_names), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(class_names), 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
ax.tick_params(which='minor', bottom=False, left=False)

ax.set_xlabel('Classe predetta')
ax.set_ylabel('Classe vera')
ax.set_title('Confusion Matrix VAEEG')

# Colorbar
plt.colorbar(im, ax=ax, label='Proporzione')
plt.savefig("sleep_results/conf_matr.png", dpi=300)
plt.show()