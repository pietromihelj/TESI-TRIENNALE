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
plt.show()