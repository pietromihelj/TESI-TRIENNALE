import argparse
import matplotlib.pylab as plt
import evaluation_functions as ef
import numpy as np
import os

parser = argparse.ArgumentParser(description='Evaluate Model')
parser.add_argument('--data_dir', type=str, required=True, help='data_dir')
parser.add_argument('--models', type=list, required=True, help='models')
parser.add_argument('--model_save_files', type=list, required=True, help='models saves files')
parser.add_argument('--params', type=list, required=True, help='params for models')
parser.add_argument('--out_dir', type=str, required=True, help='out_dir')
parser.add_argument('--cuts', type=list, required=False, help='cuts', default=[0,-1])

opts = parser.parse_args()

data_dir = opts.data_dir
models = opts.models
model_save_files = opts.model_save_files
param_list = opts.params
out_dir = opts.out_dir
cuts = opts.cuts

Bands = ['delta','theta', 'alpha', 'low_beta', 'high_beta']

pcc_con_maes = []
pvl_maes = []
pc_maes = []
pearson_maes = []
nrmse_maes = []

for model, files,params in zip(models, model_save_files, param_list):
    pcc_con_mae, pvl_mae, pc_mae, pearson_mae, nrmse_mae = ef.evaluate(data_dir=data_dir, model=model, model_files=files,params=params, out_dir=out_dir, cuts=cuts, graphs=True)
    pcc_con_maes.append(pcc_con_mae)
    pvl_maes.append(pvl_mae)
    pc_maes.append(pc_mae)
    pearson_maes.append(pearson_maes)
    nrmse_maes.append(nrmse_mae)

values = np.array([pcc_con_maes, pvl_maes, pc_maes])
values_names = ['Correlation Mean Absolute Error', 'PVL Mean Absolute Error', 'Phase Mean Absolute Error']
y_labs = ['Correlation MAE', 'PVL MAE', 'Phase MAE[rad]']

fig, axes = plt.subplot(1,3,figsize=(6*3,3))
for j in range(3):
    ax = axes[j]
    vals = values[j].T
    x = np.arange(len(Bands))
    width = 0.15
    
    for i in range(len(models)):
        plt.bar(x+i*width, vals[:,i], width, label=models[i])
    
    plt.xticks(x + 2*width, Bands)  # centratura delle bande
    plt.ylabel(y_labs[j])
    plt.title(values_names[j])
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir,f'{y_labs[j]}.png')
    plt.savefig(path)
    plt.show()

y_labs = ['Pearson correlation', 'NRMSE']
titles = ['Pearson correlation Coefficient', 'NRMSE']
fig1, axes1 = plt.subplot(1,2, figsize=(3*2, 3))
for i in range(2):
    ax = axes[i]
    x = np.arange(len(models))

    plt.bar(x, pearson_maes, width)

    plt.xticks(models)
    plt.ylabel(y_labs[i])
    plt.title(titles[i])
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f'{y_labs[i]}.png')
    plt.savefig(path)
    plt.show()
