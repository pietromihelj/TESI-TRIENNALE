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
parser.add_argument('--model', type=str, required=False, help='model', default=None)
parser.add_argument('--model_save_name', type=str, required=False, help='model_save_name', default=None)
parser.add_argument('--model_save_files', nargs='+', type=str, required=False, help='models saves files', default=None)
parser.add_argument('--params', nargs='+',type=str, required=False, help='params for models', default=[])
parser.add_argument('--out_dir', type=str, required=True, help='out_dir')
parser.add_argument('--cuts', nargs='+',type=int, required=False, help='cuts', default=[0,-1])

opts = parser.parse_args()

data_dir = opts.data_dir
model = opts.model
model_save_name = opts.model_save_name
model_save_files = opts.model_save_files
param_list = [[int(x) for x in group] for group in parse_list_of_lists(opts.params, 'f')]
out_dir = opts.out_dir
cuts = opts.cuts

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print(f'EVALUATION DEL MODELLO: {model_save_name}')
pcc_con_mae, pvl_mae, pc_mae, pearson_mae, nrmse_mae, or_ampl, rec_ampl = ef.evaluate(data_dir=data_dir, model=model, model_files=model_save_files,params=param_list, cuts=cuts)
t, p = ttest_rel(or_ampl,rec_ampl)

print('pcc_con_mae: ',pcc_con_mae)
print('pvl_mae', pvl_mae)
print('pc_maes: ',pc_mae)
print('pearson_maes: ', pearson_mae)
print('nrmse_mae: ', nrmse_mae)


values = [pearson_mae, nrmse_mae, pc_mae, pcc_con_mae, pvl_mae]
titles = ['Pearson CC', 'NRMSE', 'Phase Mean Absolute Error', 'Correlation Mean Absolute Error', 'PVL Mean Absolute Error']
y_lab = ['PCC', 'NRMSE', 'Phase MAE', 'Correlation MAE', 'PVL MAE']

df = pd.DataFrame({'metrics': values+[p,t]}, index = y_lab+['Mean Amplitude Difference', 'p-value'])
save_path = os.path.join(out_dir, f'metrics_saves_{model_save_name}.csv')
df.to_csv(save_path)
