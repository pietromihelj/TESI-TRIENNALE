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