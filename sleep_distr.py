import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Caricamento dati ---
eeg_W = np.load("sleep_tesat_data/11089_15991.npy")
eeg_R = np.load("sleep_tesat_data/11086_3460.npy")
eeg_N3 = np.load("sleep_tesat_data/11083_21130.npy")
eeg_N2 = np.load("sleep_tesat_data/11080_2014.npy")
eeg_N1 = np.load("sleep_tesat_data/11074_18238.npy")

eegs = [eeg_N1, eeg_N2, eeg_N3, eeg_R, eeg_W]

# --- Pulizia clip corrotte ---
inf_clip = 0
inf_num_clip = []
for eeg in eegs:
    for clip in eeg:
        if np.any(np.isinf(clip)):
            inf_clip += 1
            inf_num_clip.append(np.sum(np.isinf(clip)))

print(f"Clip corrotte: {inf_clip}. Inf medi per clip: {np.mean(inf_num_clip) if inf_num_clip else 0}")

clean_eegs = []
for eeg in eegs:
    clean_eegs.append(np.array([clip for clip in eeg if not np.any(np.isinf(clip))]))

print(clean_eegs[4].shape)

# --- Suddivisione bande ---
deltas, thetas, alphas, low_betas, high_betas = [], [], [], [], []
for clean_eeg in clean_eegs:
    deltas.append(clean_eeg[:, :8])        # delta = prime 8 colonne
    thetas.append(clean_eeg[:, 8:18])      # theta = colonne 8-17
    alphas.append(clean_eeg[:, 18:30])     # alpha = colonne 18-29
    low_betas.append(clean_eeg[:, 30:40])  # low beta = colonne 30-39
    high_betas.append(clean_eeg[:, 40:])   # high beta = colonne 40 in poi

bands = {
    "delta": deltas,
    "theta": thetas,
    "alpha": alphas,
    "low_beta": low_betas,
    "high_beta": high_betas,
}

# --- PCA ---
pca = PCA(2)
bands_pca = {band: [] for band in bands}

for band, eeg_list in bands.items():
    # Fit PCA su tutti i dati della banda
    all_data = np.concatenate(eeg_list, axis=0)
    pca.fit(all_data)

    # Trasforma ogni sleep stage separatamente
    for eeg_stage in eeg_list:
        bands_pca[band].append(pca.transform(eeg_stage))

# --- Plot distribuzioni ---
sleep_stages = ["N1", "N2", "N3", "R", "W"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

fig, axes = plt.subplots(5, 10, figsize=(30, 15), sharey=True)
fig.suptitle("Distribuzioni PCA per bande e sleep stages", fontsize=20)

for row, (band, stage_list) in enumerate(bands_pca.items()):
    for dim in [0, 1]:  # due componenti PCA
        for col, stage_name in enumerate(sleep_stages):
            ax = axes[row, dim*5 + col]
            data = stage_list[col][:, dim]
            ax.hist(data, bins=50, density=True, alpha=0.6, color=colors[col])
            if row == 0:
                ax.set_title(f"{stage_name} - PC{dim+1}")
        axes[row, 0].set_ylabel(band, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()