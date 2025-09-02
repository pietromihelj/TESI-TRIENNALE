import argparse
import matplotlib.pylab as plt
import evaluation_functions as ef
import numpy as np
import os
import warnings
import pandas as pd
from scipy.stats import ttest_rel
import mne
import matplotlib

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def parse_list_of_lists(flat_list, separator):
    result = []
    current_list = []
    for item in flat_list:
        if item == separator:
            if current_list:
                result.append(current_list)
                current_list = []
        else:
            current_list.append(item)
    if current_list:
        result.append(current_list)
    return result

parser = argparse.ArgumentParser(description='Evaluate Model')
parser.add_argument('--data_dir', type=str, required=True, help='data_dir')
parser.add_argument('--models', nargs='+',type=str, required=True, help='models')
parser.add_argument('--model_save_files',nargs='+', type=str, required=True, help='models saves files')
parser.add_argument('--params', nargs='+',type=int, required=False, help='params for models', default=[])
parser.add_argument('--out_dir', type=str, required=True, help='out_dir')
parser.add_argument('--cuts', nargs='+',type=int, required=False, help='cuts', default=[0,-1])

opts = parser.parse_args()

data_dir = opts.data_dir
models = opts.models
model_save_files = opts.model_save_files
param_list = parse_list_of_lists(opts.params, ',')
out_dir = opts.out_dir
cuts = opts.cuts

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

pcc_con_maes = []
pvl_maes = []
pc_maes = []
pearson_maes = []
nrmse_maes = []

for model, files,params in zip(models, model_save_files, param_list):
    print(f'EVALUATION DEL MODELLO: {model}')
    pcc_con_mae, pvl_mae, pc_mae, pearson_mae, nrmse_mae, or_ampl, rec_ampl = ef.evaluate(data_dir=data_dir, model=model, model_files=files,params=params, cuts=cuts)
    t, p = ttest_rel(or_ampl,rec_ampl)
    pcc_con_maes.append(pcc_con_mae)
    pvl_maes.append(pvl_mae)
    pc_maes.append(pc_mae)
    pearson_maes.append(pearson_mae)
    nrmse_maes.append(nrmse_mae)

print('pcc_con_mae: ',pcc_con_maes)
print('pvl_mae', pvl_maes)
print('pc_maes: ',pc_maes)
print('pearson_maes: ', pearson_maes)
print('nrmse_mae: ', nrmse_maes)


values = [pearson_maes, nrmse_maes, pc_maes, pcc_con_maes, pvl_maes]
titles = ['Pearson CC', 'NRMSE', 'Phase Mean Absolute Error', 'Correlation Mean Absolute Error', 'PVL Mean Absolute Error']
y_lab = ['PCC', 'NRMSE', 'Phase MAE', 'Correlation MAE', 'PVL MAE']

df = pd.DataFrame(values+[t,p], index = y_lab['Mean Amplitude Difference', 'p-value'], columns=['metrics'])
save_path = os.path.join(out_dir, f'metrics_saves_{models}.csv')
df.to_csv(save_path)

ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']
info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)
evoked_orig = mne.EvokedArray(or_ampl[:, np.newaxis], info)
evoked_rec = mne.EvokedArray(rec_ampl[:,np.newaxis], info)

fig_map ,axes = plt.subplots(1,2, figsize=(3,6))
im_o, cn_o = mne.viz.plot_topomap(evoked_orig.data[:, 0], evoked_orig.info,axes=axes[0],show=False,cmap='jet',contours=6,sensors=True)
im_r, cn_r = mne.viz.plot_topomap(evoked_rec.data[:, 0], evoked_rec.info,axes=axes[1],show=False,cmap='jet',contours=6,sensors=True)
for coll in axes[0].collections:
    print(type(coll))
    if isinstance(coll, matplotlib.collections.PathCollection):  # solo scatter
        coll.set_sizes([10])  # aumenta la dimensione dei marker
        coll.set_facecolor('k')            # colore pieno (nero)
        coll.set_edgecolor('k')            # bordo nero
        coll.set_alpha(1.0)
for coll in axes[1].collections:
    print(type(coll))
    if isinstance(coll, matplotlib.collections.PathCollection):  # solo scatter
        coll.set_sizes([10])  # aumenta la dimensione dei marker
        coll.set_facecolor('k')            # colore pieno (nero)
        coll.set_edgecolor('k')            # bordo nero
        coll.set_alpha(1.0)
axes[0].set_title('Ampiezza media originale')
axes[1].set_title('Ampiezza media ricostruzioni')
fig_map.colorbar(im_o, ax=axes, orientation='vertical', fraction=0.05, pad=0.05, label='Ampiezza', format='%.1f')
plt.savefig(os.path.join(out_dir, 'amplitude_comparison.png'), dpi=400)
plt.show()


fig, axs = plt.subplots(2, 3, figsize=(12, 10))
axs = axs.flatten()
x = np.arange(len(models))
for j, (vals, ax) in enumerate(zip(values, axs)):
    ax.bar(x, vals, color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(y_lab[j])
    ax.set_title(titles[j])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'metrics_comparison.png'), dpi=400)
plt.show()