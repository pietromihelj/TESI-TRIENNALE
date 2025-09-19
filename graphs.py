"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Leggi CSV
df = pd.read_csv('age_reg/VAEEG_pearson_latent.csv', index_col=0)
df_abs = df.abs().round(2)
annot = df_abs.where(df_abs >= 0.1).astype(str)
mask = df_abs < 0.2
df_abs.columns = [col.split("_")[1] for col in df_abs.columns]
annot.columns = df_abs.columns
mask.columns = df_abs.columns
plt.figure(figsize=(20, 8))
ax = sns.heatmap(
    df_abs,
    annot=annot,
    fmt="",
    cmap="Oranges",
    annot_kws={"size": 8},
    mask=mask,
    vmin=0.2,
    vmax=0.6,
    cbar=False
)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('bottom')
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', length=0, rotation=0)
plt.ylabel('Channels', fontsize=14)
plt.xlabel('Latent Feature', fontsize=14)
plt.savefig("age_reg/age_vaeeg_correlazione.png", dpi=300, bbox_inches="tight")
plt.show()


# Leggi CSV
df = pd.read_csv('age_reg/VAEEG_pearson_latent.csv', index_col=0)
df_abs = df.abs().round(2)
df_abs.columns = [col.split("_")[1] for col in df_abs.columns]
df_mean = df_abs.mean(axis=0)
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
breaks = [8, 18, 30, 40]
start = 0
colors = ['#b6a5bc', '#a4c8f0', '#a0d180', '#dbc388', "#3c74e4"]
for i, end in enumerate(breaks):
    ax.plot(df_mean.index[start:end], df_mean.values[start:end], marker='o', color=colors[i])
    start = end
ax.plot(df_mean.index[start:], df_mean.values[start:], marker='o', color=colors[len(breaks)])
ax.set_ylabel('Pearson medio', fontsize=14)
ax.set_xlabel('')
ax.tick_params(left=True, right=False, bottom=False, top=False, labelleft=True, labelbottom=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylim(0, 0.6)
ax.set_xlim(left=-0.5)
plt.savefig("age_reg/age_lat_avg_corr.png", dpi=300, bbox_inches="tight")
plt.show()


# Leggi CSV
df = pd.read_csv('age_reg/VAEEG_pearson_latent.csv', index_col=0)
df_abs = df.abs().round(2)
df_mean_per_channel = df_abs.mean(axis=1)
plt.figure(figsize=(3, 8)) 
ax = plt.gca()
plt.plot(df_mean_per_channel.values, df_mean_per_channel.index, marker='o', color='grey')
plt.xlabel('Pearson medio', fontsize=14)
plt.xlim(0.10, 0.25) # Limita l'asse X
plt.xticks([0.10, 0.17, 0.25])
ax.tick_params(axis='y', which='both', left=False, labelleft=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig("age_reg/age_ch_avg_corr.png", dpi=300, bbox_inches="tight")
plt.show()
"""

"""
import matplotlib.pyplot as plt

# Valori MAE
methods = ["FAST", "Kernel PCA", "PCA", "VAEEG"]
mae_values = [14.848131349050746,
              14.731916680640278,
              14.848131349089169,
              14.865064478435318]

# Creazione grafico
plt.figure(figsize=(7,5))
bars = plt.bar(methods, mae_values, width=0.4)  # barre più strette

# Etichette sopra le barre
for bar, val in zip(bars, mae_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.2f}", ha='center', va='bottom', fontsize=10)

# Abbellimenti
plt.ylabel("MAE")
plt.title("Age Regression - MAE Test set")
plt.ylim(14, 15.5)

# Rimuovi bordi sopra e destra
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.savefig('age_reg/MAE_comparison.png', dpi=300)
plt.show()
"""

"""
ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
sfreq = 250

# Caricamento dati
data_orig = np.load("D:/eval_latent/orig/aaaaaayx_s002_t000.npy", allow_pickle=True).squeeze()[:, 25000:25250]
data_rec = np.load("D:/eval_latent/rec/aaaaaayx_s002_t000.npy", allow_pickle=True).squeeze()[:, 25000:25250]

print(data_orig.shape)

# Info e montaggio
ch_types = ['eeg'] * len(ch_names)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Calcolo PCC per ciascun canale
pccs = np.array([pearsonr(data_orig[i], data_rec[i])[0] for i in range(data_orig.shape[0])])
fig, ax = plt.subplots(figsize=(6, 6))

im, cn = mne.viz.plot_topomap(pccs, info, axes=ax, show=False, contours=6, cmap='jet')
cbar = fig.colorbar(im, ax=fig.axes[0], shrink=0.7)  # La colorbar si riferisce solo a questa figura
cbar.set_label('PCC', rotation=90, labelpad=15)

# Salvataggio in PNG
fig.savefig("results/topmap_pcc.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)
##################################################################################################################################

data_orig = np.load("D:/eval_latent/orig/aaaaaayx_s002_t000.npy", allow_pickle=True).squeeze()[:, 25000:25250]
print(data_orig.shape)

# Info e montaggio
ch_types = ['eeg'] * len(ch_names)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Calcolo delle ampiezze medie
amplitudes = np.mean(np.abs(data_orig), axis=1)

# Creazione figura e asse
fig, ax = plt.subplots(figsize=(6, 6))

# Plot topomap con colorbar limitata alla mappa
im, cn = mne.viz.plot_topomap(amplitudes, info, axes=ax, show=False, contours=6, cmap='jet')
cbar = fig.colorbar(im, ax=fig.axes[0], shrink=0.7)  # usa l'asse della mappa
cbar.set_label('Amplitude [uV]', rotation=90, labelpad=15, fontsize=20) 

# Salvataggio in PNG
plt.title('Original Mean \n Amplitude', fontsize = 30)
fig.savefig("results/orig.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)
##########################################################################################################################################
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
sfreq = 250

# Caricamento dati
data_orig = np.load("D:/eval_latent/orig/aaaaaayx_s002_t000.npy", allow_pickle=True).squeeze()[:, 25000:25250]
data_rec = np.load("D:/eval_latent/rec/aaaaaayx_s002_t000.npy", allow_pickle=True).squeeze()[:, 25000:25250]

print(data_orig.shape)

# Info e montaggio
ch_types = ['eeg'] * len(ch_names)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Calcolo PCC per ciascun canale
nrmse = np.zeros(len(ch_names))
for i in range(len(ch_names)):
    mse = np.mean((data_orig[i] - data_rec[i])**2)
    rmse = np.sqrt(mse)
    range_orig = np.max(data_orig[i]) - np.min(data_orig[i])
    nrmse[i] = rmse / range_orig
fig, ax = plt.subplots(figsize=(6, 6))

pccs = np.array([pearsonr(data_orig[i], data_rec[i])[0] for i in range(data_orig.shape[0])])
im, cn = mne.viz.plot_topomap(nrmse, info, axes=ax, show=False, contours=6, cmap='jet')
im.set_clim(0, 1) 
cbar = fig.colorbar(im, ax=fig.axes[0], shrink=0.7)  # La colorbar si riferisce solo a questa figura
cbar.set_label('nrmse', rotation=90, labelpad=15)

# Salvataggio in PNG
plt.title('NRMSE', fontsize=30)
fig.savefig("results/topmap_nrmse.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)
#######################################################################################################################
data_orig = np.load("D:/eval_latent/orig/aaaaabuv_s002_t000.npy", allow_pickle=True).squeeze()[:, 9000:11500]
data_rec = np.load("D:/eval_latent/rec/aaaaabuv_s002_t000.npy", allow_pickle=True).squeeze()[:, 9000:11500]

n_channels, n_samples = data_orig.shape
ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels), sharex=True)

for i in range(n_channels):
    ax = axes[i]
    ax.plot(data_orig[i], label='Original', color='blue')
    ax.plot(data_rec[i], label='Reconstructed', color='red', alpha=0.7)
    ax.set_ylabel(ch_names[i], rotation=0, labelpad=40, va='center')  # label canale a sinistra
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Sample Index')
plt.tight_layout()
plt.show()

###############################################################################################################

ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
sfreq = 250

data_orig = np.load("D:/eval_latent/orig/aaaaabuv_s002_t000.npy", allow_pickle=True).squeeze()[:, 9000:11500]
print(data_orig.shape)

# Info e montaggio
ch_types = ['eeg'] * len(ch_names)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Calcolo delle ampiezze medie
amplitudes = np.mean(np.abs(data_orig), axis=1)

# Creazione figura e asse
fig, ax = plt.subplots(figsize=(6, 6))

# Plot topomap con colorbar limitata alla mappa
im, cn = mne.viz.plot_topomap(amplitudes, info, axes=ax, show=False, contours=6, cmap='jet')
cbar = fig.colorbar(im, ax=fig.axes[0], shrink=0.7)  # usa l'asse della mappa
cbar.set_label('Amplitude [uV]', rotation=90, labelpad=15, fontsize=20) 

# Salvataggio in PNG
plt.title('Original Mean \n Amplitude', fontsize = 30)
fig.savefig("results/orig.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

#########################################################################################################

n_channels, n_samples = data_orig.shape
ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

sfreq = 250  # campioni al secondo

# Spacing verticale tra i canali
offset = 200
y_offsets = np.arange(n_channels)[::-1] * offset

fig, ax = plt.subplots(figsize=(15, 10))

for i in range(n_channels):
    ax.plot(data_orig[i] + y_offsets[i], color='black')
    ax.plot(data_rec[i] + y_offsets[i], color="#ccb822", linestyle='--', alpha=0.7)
    
    # Nome canale a sinistra
    ax.text(-50, y_offsets[i], ch_names[i], va='center', ha='right', fontsize=10)

# Spine e ticks
for side in ['top', 'right', 'bottom', 'left']:
    ax.spines[side].set_visible(True)
ax.tick_params(left=False, right=False, bottom=True, labelleft=False)

# Limiti asse X in campioni
ax.set_xlim(0, n_samples)

# Tick asse X in secondi
n_seconds = n_samples / sfreq
xticks_sec = np.arange(0, n_seconds+1, 1)  # 1 secondo in 1 secondo
xticks_samples = xticks_sec * sfreq
ax.set_xticks(xticks_samples)
ax.set_xticklabels([str(int(s)) for s in xticks_sec])

ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('results/channel_comparison_single_ax_labels_xlim_sec.png', dpi=300)
plt.show()

####################################################################################################

import numpy as np
from mne.filter import filter_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib import cm

data_orig = np.load("D:/eval_latent/orig/aaaaabuv_s002_t000.npy", allow_pickle=True).squeeze()[:, 9000:11500]
data_rec = np.load("D:/eval_latent/rec/aaaaabuv_s002_t000.npy", allow_pickle=True).squeeze()[:, 9000:11500]
data_orig = data_orig.astype(np.float64)
data_rec = data_rec.astype(np.float64)
ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
sfreq = 250

_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "low_beta": (13, 20),
    "high_beta": (20, 30.0)
}

corr_orig = {}
corr_rec = {}

for band, (fmin, fmax) in _BANDS.items():
    # Filtra i dati (dimensione: canali x samples)
    filtered_orig = filter_data(data_orig, sfreq, fmin, fmax, method='iir', verbose=False)
    filtered_rec = filter_data(data_rec, sfreq, fmin, fmax, method='iir', verbose=False)
    
    # Calcola matrice di correlazione tra canali
    corr_orig[band] = np.corrcoef(filtered_orig)
    corr_rec[band] = np.corrcoef(filtered_rec)
    
    print(f"Band {band}: Corr matrix shape = {corr_orig[band].shape}")

# Plot

fig = plt.figure(figsize=(20, 10))

bands = list(_BANDS.keys())
n_bands = len(bands)
axes = []

# Larghezza e altezza di ciascuna heatmap
width = 0.16
height = 0.38  # altezza per ciascuna riga

# Prepara colormap con NaN bianchi
cmap = cm.get_cmap('jet').copy()
cmap.set_bad(color='white')

for i, band in enumerate(bands):
    # Coordinate x della colonna
    left = 0.05 + i * (width + 0.015)
    
    # Originale in alto
    bottom = 0.55
    ax = fig.add_axes([left, bottom, width, height])
    mat = corr_orig[band].copy()
    np.fill_diagonal(mat, np.nan)  # diagonale bianca
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap=cmap)
    
    if i == 0:
        ax.set_ylabel("Original", rotation=90, labelpad=15, fontsize=12)
    ax.set_xticks(range(len(ch_names)))
    ax.set_yticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_yticklabels(ch_names, fontsize=6)
    axes.append(ax)
    
    # Ricostruito in basso
    bottom = 0.2
    ax = fig.add_axes([left, bottom, width, height])
    mat = corr_rec[band].copy()
    np.fill_diagonal(mat, np.nan)  # diagonale bianca
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap=cmap)
    
    if i == 0:
        ax.set_ylabel("Reconstructed", rotation=90, labelpad=15, fontsize=12)
    ax.set_xticks(range(len(ch_names)))
    ax.set_yticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_yticklabels(ch_names, fontsize=6)
    axes.append(ax)

# Bande in alto
for i, band in enumerate(bands):
    axes[i*2].set_title(band, fontsize=12)

# Colorbar a destra
cbar_ax = fig.add_axes([0.92, 0.22, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Connettività', fontsize=15)

plt.savefig('results/connettivity_matr.png', dpi=300, bbox_inches='tight')
plt.show()
#######################################################################################
import numpy as np
from mne.filter import filter_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.ticker import  FormatStrFormatter

metrics = ["PCC", "NRMSE", "Phase MAE", "Correlation MAE", "PVL MAE"]

baseline_pca = [0.01673942251439155, 249983.55542442997, 1.5128213170685327, 0.6217784182361765, 0.47963429876926933]
baseline_ica = [0.01673942251439155, 249983.55542442997, 1.5128213170685327, 0.6217784182361765, 0.47963429876926933]
vaeeg = [0.39691917101542157, 12218028032.0, 0.3259616792201996, 0.09518110007047653, 0.019895992144625823]

methods = ["PCA", "FastICA", "VAEEG"]
data = np.array([baseline_pca, baseline_ica, vaeeg])
width = 0.18

fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)

for i, metric in enumerate(metrics):
    ax = axes[i]
    x = np.arange(len(methods))
    ax.bar(x, data[:, i], width, color=["#445058", "#3b3631", "#384538"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, ha='center')
    ax.set_title(metric)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Solo per NRMSE: tick con notazione scientifica
    if metric == "NRMSE":
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

fig.suptitle("Confronto delle metriche tra Baseline e VAEEG", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/metric_comparison.png', dpi=300)
plt.show()
 ###################################################################
import matplotlib.pyplot as plt
import numpy as np

metrics = {
    "PCC": 0.01673942251439155,
    "NRMSE": 249983.55542442997,
    "Phase MAE": 1.5128213170685327,
    "Correlation MAE": 0.6217784182361765,
    "PVL MAE": 0.47963429876926933
}

models = ["P_lcosh", "P_cube", "P_exp", "D_losh", "D_cube", "D_exp"]

# Crea figura con 5 subplot orizzontali
fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 1 riga, 5 colonne

# Scala di grigi per 6 modelli
colors = plt.cm.Greys(np.linspace(0.3, 0.9, len(models)))

for ax, (metric_name, value) in zip(axes, metrics.items()):
    values = [value] * len(models)
    ax.bar(models, values, color=colors)
    
    # Titolo e label
    ax.set_title(metric_name)
    ax.set_ylim(0, max(values)*1.2)
    
    # Rimuovi righe in alto e a sinistra
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Etichette x ruotate per leggibilità
    ax.set_xticklabels(models, rotation=45, ha='right')

plt.tight_layout()
plt.savefig("results/baseline_comparison.png")
plt.show()

"""

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dati combinati dei 4 CSV
data = {
    'Embedding': ['KernelPCA']*3 + ['PCA_0']*3 + ['VAEEG']*3 + ['FastICA_0']*3,
    'Predictor': ['LinearRegression','Lasso','Ridge']*4,
    'MAE': [14.731916680640278, 14.731916680640278, 14.731916680640278,
            14.848131349089169, 14.84798044732484, 14.848131349089135,
            14.865064478435318, 18.522320235827124, 21.630034310003985,
            14.848131349050746, 14.847979112163102, 14.848131349033821]
}

df = pd.DataFrame(data)

plt.figure(figsize=(15,5))
sns.set_style("white")  # elimina la griglia di sfondo

# Barplot affiancato con scala di grigi
ax = sns.barplot(x="Predictor", y="MAE", hue="Embedding", data=df, palette="Greys")

# Rimuovi i bordi superiori e destri
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Aggiungi etichette
ax.set_ylabel("MAE", fontsize=14)
ax.set_xlabel("Predittore", fontsize=14)
ax.set_title("Valutazione performance: MAE", fontsize=16)
plt.legend(title="Embedding")

plt.savefig('age_reg/Age_comp_perf.png', dpi=300, bbox_inches='tight')
plt.show()"""

"""

from utils import get_path_list
import numpy as np
from tqdm import tqdm

print("Caricamento dati")

normal_dir = "C:/Users/Pietro/Desktop/archive/VAEEG/normal"
seizure_dir = "C:/Users/Pietro/Desktop/archive/VAEEG/seizure"
normal_paths = get_path_list(normal_dir, f_extensions=['.npy'], sub_d=True)
seizure_paths = get_path_list(seizure_dir, f_extensions=['.npy'], sub_d=True)

clips_norm = []
print('normal processing')
for path in tqdm(normal_paths):
    normal_eeg = np.load(path, allow_pickle=True)

    finite_vals = normal_eeg[np.isfinite(normal_eeg)]
    max_val = finite_vals.max()
    min_val = finite_vals.min()

    # sostituisci inf
    normal_eeg[np.isposinf(normal_eeg)] = max_val
    normal_eeg[np.isneginf(normal_eeg)] = min_val

    # sostituisci nan con media dei valori finiti
    nan_mask = np.isnan(normal_eeg)
    normal_eeg[nan_mask] = np.mean(finite_vals)

    clips_norm.append(np.mean(normal_eeg, axis=0))
clips_norm = np.concatenate(clips_norm)

clips_seiz = []
print('seizure processing')
for path in tqdm(seizure_paths):
    seiz_eeg = np.load(path, allow_pickle=True)

    finite_vals = seiz_eeg[np.isfinite(seiz_eeg)]
    max_val = finite_vals.max()
    min_val = finite_vals.min()

    seiz_eeg[np.isposinf(seiz_eeg)] = max_val
    seiz_eeg[np.isneginf(seiz_eeg)] = min_val

    nan_mask = np.isnan(seiz_eeg)
    seiz_eeg[nan_mask] = np.mean(finite_vals)

    clips_seiz.append(np.mean(seiz_eeg, axis=0))
clips_seiz = np.concatenate(clips_seiz)




print('Distribution comparison')
distributions = {}
for i in range(50):
    latent_norm = clips_norm[:, i]
    latent_seiz = clips_seiz[:, i]
    distributions[f'latent_{i}'] = {'normal': latent_norm, 'seizure': latent_seiz}

from scipy.stats import mannwhitneyu
p_values = []
for i in range(50):
    norm_vals = distributions[f'latent_{i}']['normal']
    seiz_vals = distributions[f'latent_{i}']['seizure']
    stat, p = mannwhitneyu(norm_vals, seiz_vals)
    print(f"Latent {i}: p-value={p}")
    p_values.append(p)
np.save('seiz_detec/p_values.npy', p_values)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

# --- Ripristina notazione matematica/scientifica globale ---
rcParams['axes.formatter.useoffset'] = True
rcParams['axes.formatter.use_mathtext'] = True
rcParams['axes.formatter.use_locale'] = False

# Latent più significativi
top_latents = [3, 7, 5, 8, 33, 36, 21, 16, 30, 20]

n_rows, n_cols = 2, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))

eps = 1e-6  # piccolissimo rumore per varianza zero

for idx, latent_idx in enumerate(top_latents):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    # Preleva le distribuzioni
    latent_norm = distributions[f'latent_{latent_idx}']['normal']
    latent_seiz = distributions[f'latent_{latent_idx}']['seizure']

    # Aggiungi rumore se varianza zero
    if np.all(latent_norm == latent_norm[0]):
        latent_norm = latent_norm + np.random.normal(0, eps, size=latent_norm.shape)
    if np.all(latent_seiz == latent_seiz[0]):
        latent_seiz = latent_seiz + np.random.normal(0, eps, size=latent_seiz.shape)

    # KDE
    sns.kdeplot(latent_norm, ax=ax, color='blue', label='normal', fill=True, alpha=0.3)
    sns.kdeplot(latent_seiz, ax=ax, color='red', label='seizure', fill=True, alpha=0.3)

    # Titolo con font più grande
    ax.set_title(f'Latent {latent_idx}', fontsize=16)  # <-- qui ingrandito
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Solo 3 tick sull'asse x
    xmin, xmax = ax.get_xlim()
    ticks = np.linspace(xmin, xmax, 3)
    ax.set_xticks(ticks)

    # Ripristina notazione matematica/scientifica su questo asse
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("seiz_detec/top10_latent_density.png", dpi=300)
plt.show()
"""

"""
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

# P-value dei latent
p_values = [
    0.9324071674036342, 0.4674278194487932, 0.8459857253941154, 6.314700480849854e-88, 0.8459857253941154,
    1.5075963339837503e-57, 0.8459857253941154, 6.31470105614695e-88, 3.503140522419382e-32, 0.866914372252015,
    0.3511175576830534, 0.09821202882715967, 0.22208934054049656, 0.007601419013093555, 0.20612431833624378,
    0.36441221821006353, 5.655153063511487e-07, 0.015456473436150706, 0.07334497306433482, 0.0027533881806481317,
    0.00010847200219653572, 7.696612321041706e-07, 0.16366342197646355, 0.0572713163621631, 0.29288915742653965,
    0.044211170076849333, 0.40846245818558125, 0.7957049333263377, 0.6330202207117746, 0.4432938811488353,
    1.133115829301052e-06, 1.424333689355922e-05, 1.3409487522062662e-05, 2.9515219223942896e-10, 5.114373134325002e-05,
    2.9758623491354074e-06, 1.7827581728451627e-07, 2.4731801529785605e-06, 0.00016029483509015909, 2.2775195326315773e-08,
    0.5803389031418555, 0.4107779718031189, 0.30784633217647783, 0.3983579519245851, 0.11514468998861713,
    0.5039748947254974, 0.22067550926915935, 0.9890383885065817, 0.029883840989796177, 0.43990523445588703
]

# Soglia significatività
alpha = 0.05
significant = np.array(p_values) < alpha

# Calcolo percentuale di variabili significative
percent_significant = np.sum(significant) / len(significant) * 100
print(f"Percentuale di variabili significative: {percent_significant:.2f}%")

# P-value dei latent
p_values = [
    0.9324071674036342, 0.4674278194487932, 0.8459857253941154, 6.314700480849854e-88, 0.8459857253941154,
    1.5075963339837503e-57, 0.8459857253941154, 6.31470105614695e-88, 3.503140522419382e-32, 0.866914372252015,
    0.3511175576830534, 0.09821202882715967, 0.22208934054049656, 0.007601419013093555, 0.20612431833624378,
    0.36441221821006353, 5.655153063511487e-07, 0.015456473436150706, 0.07334497306433482, 0.0027533881806481317,
    0.00010847200219653572, 7.696612321041706e-07, 0.16366342197646355, 0.0572713163621631, 0.29288915742653965,
    0.044211170076849333, 0.40846245818558125, 0.7957049333263377, 0.6330202207117746, 0.4432938811488353,
    1.133115829301052e-06, 1.424333689355922e-05, 1.3409487522062662e-05, 2.9515219223942896e-10, 5.114373134325002e-05,
    2.9758623491354074e-06, 1.7827581728451627e-07, 2.4731801529785605e-06, 0.00016029483509015909, 2.2775195326315773e-08,
    0.5803389031418555, 0.4107779718031189, 0.30784633217647783, 0.3983579519245851, 0.11514468998861713,
    0.5039748947254974, 0.22067550926915935, 0.9890383885065817, 0.029883840989796177, 0.43990523445588703
]

# Soglia significatività
alpha = 0.05
significant = np.array(p_values) < alpha

# Griglia 5x10
n_rows, n_cols = 5, 10
fig, ax = plt.subplots(figsize=(10, 5))

# Creiamo una matrice dei colori
colors = np.where(significant.reshape(n_rows, n_cols), 'orange', 'white')

# Disegniamo le celle
for i in range(n_rows):
    for j in range(n_cols):
        latent_idx = i * n_cols + j
        rect = plt.Rectangle((j, n_rows - i - 1), 1, 1, color=colors[i, j], ec='black')
        ax.add_patch(rect)
        ax.text(j + 0.5, n_rows - i - 0.5, f'Latent {latent_idx}', ha='center', va='center', fontsize=10, color='black')

ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_title('Significatività variabili latenti (p<0.05)', fontsize=16)
plt.tight_layout()
plt.savefig("sleep_results/significative_matr.png")
plt.show()
"""

"""
import matplotlib.pyplot as plt
import numpy as np

# Modelli
models = ['VAEEG', 'PCA', 'FAStICA', 'KernelPCA']

# Metriche reali
test_accuracy = [0.9689999999999999] * 4
recall_norm = [1.0] * 4
recall_seiz = [0.0] * 4

metrics = [test_accuracy, recall_norm, recall_seiz]
metric_names = ['Test Accuracy', 'Recall Normal', 'Recall Seizure']

# Posizione delle barre
x = np.arange(len(models))
width = 0.6  # barre più strette

# Toni di grigio
colors = ['#d9d9d9', '#b0b0b0', '#707070', '#303030']

# Crea figura con 3 subplot orizzontali
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

for i, ax in enumerate(axes):
    bars = ax.bar(x, metrics[i], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric_names[i])
    ax.set_title(f'{metric_names[i]} per modello', fontsize=14)
    
    # Aggiungi valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Rimuovi bordo superiore e destro
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("seiz_detec/metrics.png")
plt.show()
"""

"""
import mne
import numpy as np
import matplotlib.pyplot as plt
from seizure_dataset_creation import parse_summary_txt
from utils import select_bipolar, to_monopolar

# --- Parametri: file EDF e summary corrispondente ---
edf_file = "C:/Users/Pietro/Desktop/archive/chb-mit-scalp-eeg-database-1.0.0/chb16/chb16_17.edf"
summary_file = "C:/Users/Pietro/Desktop/archive/chb-mit-scalp-eeg-database-1.0.0/chb16/chb16-summary.txt"

# --- Parsing del file di summary ---
_, df = parse_summary_txt(summary_file)

# --- Filtra solo la riga relativa all’EDF selezionato ---
df_file = df[df['File Name'] == edf_file.split("/")[-1]]

if df_file.empty:
    raise ValueError(f"Nessuna riga trovata in {summary_file} per {edf_file}")

# --- Funzione per la seizure più corta ---
def min_seizure_duration(row):
    starts = row['Seizure Start Times']
    ends = row['Seizure End Times']
    if len(starts) == 0 or len(ends) == 0 or len(starts) != len(ends):
        return np.inf
    durations = np.array(ends) - np.array(starts)
    return durations.min()

df_file['min_seizure_duration'] = df_file.apply(min_seizure_duration, axis=1)

# --- Riga con la seizure più corta in questo file ---
idx_min = df_file['min_seizure_duration'].idxmin()
row_min = df_file.loc[idx_min]

print("Seizure più corta in questo file:")
print(f"File Name: {row_min['File Name']}")
print(f"Durata minima: {row_min['min_seizure_duration']} secondi")

# --- Carica EEG ---
def has_monopolar(ch_names, sep='-'):
    for ch in ch_names:
        if sep not in ch:
            return True
    return False
print(f"Caricamento EDF: {edf_file}")
eeg = mne.io.read_raw_edf(edf_file, preload=True)
if not has_monopolar(eeg.info['ch_names']):
    eeg, _ = select_bipolar(eeg)
    eeg, _ = to_monopolar(eeg, 'CZ')

# --- Trova la seizure più corta nello slice ---
starts = np.array(row_min['Seizure Start Times'])
ends   = np.array(row_min['Seizure End Times'])
durations = ends - starts
idx_shortest = durations.argmin()

sfreq = eeg.info['sfreq']
start_idx = int(starts[idx_shortest] * sfreq)
end_idx   = int(ends[idx_shortest] * sfreq)

# --- Slice: 15 secondi prima e dopo ---
slice_start = max(int(start_idx - 15*sfreq), 0)
slice_end   = min(int(end_idx + 15*sfreq), eeg.n_times)

# --- Estrai dati EEG ---
data_eeg = eeg.get_data()[:, slice_start:slice_end] * 1e6
n_channels, n_samples = data_eeg.shape
print(f"Shape dei dati EEG: {data_eeg.shape}")

# --- Nomi canali ---
ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

# --- Offset verticale ---
offset = 200
y_offsets = np.arange(n_channels)[::-1] * offset

# --- Identifica eventuali altre seizure nello slice ---
starts_samples = starts * sfreq
ends_samples   = ends * sfreq
mask_in_slice = (ends_samples > slice_start) & (starts_samples < slice_end)
other_seizures_in_slice = list(zip(starts_samples[mask_in_slice], ends_samples[mask_in_slice]))

# --- Plot ---
plt.figure(figsize=(15, 8))
for i in range(n_channels):
    plt.plot(np.arange(n_samples)/sfreq, data_eeg[i]+y_offsets[i], color='blue')
    plt.text(-0.1, y_offsets[i], ch_names[i], va='center', ha='right', fontsize=10)

# Evidenzia la seizure più corta
plt.axvspan((start_idx - slice_start)/sfreq, (end_idx - slice_start)/sfreq, 
            color='red', alpha=0.3, label='Seizure più corta')

# Evidenzia altre seizure nello slice
for s_start, s_end in other_seizures_in_slice:
    if s_start == start_idx and s_end == end_idx:
        continue
    plt.axvspan((s_start - slice_start)/sfreq, (s_end - slice_start)/sfreq, 
                color='red', alpha=0.15, label='Altra seizure')

plt.xlabel("Tempo (s)")
plt.yticks([])
plt.tight_layout()
plt.savefig("seiz_detec/seizure_shortest_in_file.png", dpi=300)
plt.show()

from deploy import get_orig_rec_latent, load_models

model = load_models(model='VAEEG', save_files=['models/VAEEG/delta_band.pth', 'models/VAEEG/theta_band.pth', 'models/VAEEG/alpha_band.pth', 'models/VAEEG/low_beta_band.pth', 'models/VAEEG/high_beta_band.pth'], params=[[8], [10], [12], [10], [10]])
pi, pj, latent_norm = get_orig_rec_latent(raw=data_eeg , model=model, fs=256)
print(latent_norm.shape)
latent_norm = latent_norm.mean(axis = 0)
print(latent_norm.shape)
print("Shape finale:", latent_norm.shape)  # (30, 50)

latent_mat = latent_norm.T  # adesso è (50, 30): latenti (righe) × clip (colonne)

plt.figure(figsize=(15, 8))  
im = plt.imshow(latent_mat, aspect='auto', origin='lower', cmap='bwr')  
cbar = plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04, shrink=0.8, location='bottom')
cbar.set_label("Valore latente", fontsize=10)
plt.xlabel("Clip (30)", fontsize=12)
plt.ylabel("Dimensione latente (50)", fontsize=12)
plt.title("Heatmap latent_norm (latenti × clip)", fontsize=14)

plt.tight_layout()
plt.savefig("seiz_detec/latent_heatmap_bwr.png", dpi=300)
plt.show()

"""

"""

import matplotlib.pyplot as plt
import numpy as np

# P-value dei latent
p_values = [
    0.9324071674036342, 0.4674278194487932, 0.8459857253941154, 6.314700480849854e-88, 0.8459857253941154,
    1.5075963339837503e-57, 0.8459857253941154, 6.31470105614695e-88, 3.503140522419382e-32, 0.866914372252015,
    0.3511175576830534, 0.09821202882715967, 0.22208934054049656, 0.007601419013093555, 0.20612431833624378,
    0.36441221821006353, 5.655153063511487e-07, 0.015456473436150706, 0.07334497306433482, 0.0027533881806481317,
    0.00010847200219653572, 7.696612321041706e-07, 0.16366342197646355, 0.0572713163621631, 0.29288915742653965,
    0.044211170076849333, 0.40846245818558125, 0.7957049333263377, 0.6330202207117746, 0.4432938811488353,
    1.133115829301052e-06, 1.424333689355922e-05, 1.3409487522062662e-05, 2.9515219223942896e-10, 5.114373134325002e-05,
    2.9758623491354074e-06, 1.7827581728451627e-07, 2.4731801529785605e-06, 0.00016029483509015909, 2.2775195326315773e-08,
    0.5803389031418555, 0.4107779718031189, 0.30784633217647783, 0.3983579519245851, 0.11514468998861713,
    0.5039748947254974, 0.22067550926915935, 0.9890383885065817, 0.029883840989796177, 0.43990523445588703
]

# Soglia significatività
alpha = 0.05
significant = np.array(p_values) < alpha

# Griglia 5x10
n_rows, n_cols = 5, 10
fig, ax = plt.subplots(figsize=(12, 6))

# Creiamo una matrice dei colori
colors = np.where(significant.reshape(n_rows, n_cols), 'orange', 'white')

# Disegniamo le celle con nome del latent e p-value a 3 cifre significative
for i in range(n_rows):
    for j in range(n_cols):
        latent_idx = i * n_cols + j
        rect = plt.Rectangle((j, n_rows - i - 1), 1, 1, color=colors[i, j], ec='black')
        ax.add_patch(rect)
        # Mostra nome latent e p-value
        p_val = p_values[latent_idx]
        ax.text(j + 0.5, n_rows - i - 0.5, f'Latent {latent_idx}\n{p_val:.3g}',
                ha='center', va='center', fontsize=9, color='black')

ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_title('Significatività variabili latenti (p < 0.05)', fontsize=16)
plt.tight_layout()
plt.savefig("sleep_results/significative_matr.png", dpi=300)
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Etichette delle classi
classes = ['N1', 'N2', 'N3', 'R', 'W']

# Valori di accuracy per classe (diagonale della matrice)
accuracies = [0.00, 1.00, 0.00, 0.05, 0.06]

# Costruisco la matrice di confusione
conf_matrix = np.zeros((5, 5))

# Imposto la diagonale con le accuratezze
np.fill_diagonal(conf_matrix, accuracies)

# Metto tutti i valori non predetti correttamente su N2 (colonna 1)
for i in range(5):
    conf_matrix[i, 1] = 1.0 - conf_matrix[i, i]  # errore = 1 - accuracy
    conf_matrix[i, i] = accuracies[i]            # rimetto la diagonale

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Classe Predetta")
plt.ylabel("Classe Reale")
plt.title("Confusion Matrix")
plt.savefig('conf_matr.png', dpi=300)
plt.show()