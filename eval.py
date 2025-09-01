import argparse
import matplotlib.pylab as plt
import evaluation_functions as ef
import numpy as np
import os
import warnings
import pandas as pd

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
    pcc_con_mae, pvl_mae, pc_mae, pearson_mae, nrmse_mae = ef.evaluate(data_dir=data_dir, model=model, model_files=files,params=params, cuts=cuts)
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

df = pd.DataFrame(values, index = y_lab, columns=['metrics']+models)
save_path = os.path.join(out_dir, f'metrics_saves_{models}.csv')
df.to_csv(save_path)

fig, axs = plt.subplots(3, 2, figsize=(12, 10))
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

fig.delaxes(axs[-1])
plt.tight_layout()
plt.savefig(out_dir)
plt.show()