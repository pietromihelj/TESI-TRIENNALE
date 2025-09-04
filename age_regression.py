from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  mean_absolute_error, r2_score
from deploy import load_models, get_orig_rec_latent
from scipy.stats import pearsonr
from utils import get_path_list, get_raw, check_channel_names
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Age Regression')
parser.add_argument('--model_names', nargs='+', type=str, required=True, help='model_names')
parser.add_argument('--model_paths', nargs='+', type=str, required=True, help='model_paths')
parser.add_argument('--params', nargs='+', type=list, required=False, help='params', default=[])
parser.add_argument('--out_dir', type=str, required=True, help='out_dir')

opts = parser.parse_args()

model_names = opts.model_names
model_paths = opts.model_paths

params = opts.params
out_dir = opts.out_dir

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
"""
Nella successiva parte di codice carico i dati e creo 2 liste contenenti la coppia eeg eta soggetto
"""
path = get_path_list("D:/nmt_scalp_eeg_dataset", f_extensions=['.edf'], sub_d=True)
age_df = pd.read_csv("D:/nmt_scalp_eeg_dataset/Labels.csv")
ages = []
raws = []
for j,p in enumerate(path):
    raw = get_raw(p)
    check_channel_names(raw_obj=raw, verbose=False)
    raws.append(raw.get_data())
    code = os.path.basename(p)
    ages.append(age_df.loc[age_df['recordname'] == code, 'age'].iloc[0])
    if j==1:
        break

print('AGES: ', len(ages))
print('RAWS: ', len(raws))

#inizzializzo i predittori di test
predictors = [
    ("LinearRegression", LinearRegression()),
    ("Lasso", Lasso(alpha=1.0)),
    ("Ridge", Ridge(alpha=1.0)),
    ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42))
]

results = {}

"""
Nella successiva parte di codice trovo il modello di embedding che funziona meglio sulla regressione delle età.
Considero una singola clip come campione. per ogni soggetto prendo come età la media delle clip
"""
#itero su ogni modello di embedding
for model_name, model_file, param in zip(model_names,model_paths, params):
    print(f'entraro nel ciclo main per il modello {model_name}')
    #per ognungo creo la stratifiedkfold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    #prendo le var latent
    model=load_models(model_name, model_file, param)
    latents = []
    for j,raw in enumerate(raws):
        _,_,latent = get_orig_rec_latent(raw, model)
        latents.append(latent.reshape(-1,50))
    print('Latenti create: ', len(latents), ' - ', latents[0].shape)
    quantiles = pd.qcut(ages, q=5, labels=False)
    partial_res = []
    #per ogni predittore eseguo la k_fold
    for name,pred in predictors:
        par_mae = []
        par_r2 = []
        for idx, (tr_idx, te_idx) in skf.split(latents, quantiles):
            #tr_X contiene gli eeg interi ed ha forma [N, k, 50]
            tr_X = latents[tr_idx]
            te_X = latents[te_idx] 
            tr_Y = ages[tr_idx]
            #creo una lista tr_Y che contiene le età associate ad ogni clip
            tr_Y = np.concatenate([np.full(eeg.shape[0], ages[j])] for j, eeg in enumerate(tr_X))
            #concateno infine le clip per ottenere una lista [N*k, 50] per usare le clip come dato 
            tr_X = np.concatenate(tr_X, axis = 0)
            #eeg_limits_idx è una lista contenente i cut da usare per ricostruire gli eeg durante il calcolo delle metriche
            eeg_limits_idx = [0]
            eeg_limits_idx = eeg_limits_idx + [eeg_limits_idx[j] + len(eeg) for j, eeg in enumerate(te_X)]
            #concateno il test per avere anche cui la clip come dato base
            te_X = np.concatenate(te_X)
            #te_Y è solo l'età dell'eeg totale
            te_Y = ages[te_idx]
            #fitto il modello
            pred.fit(tr_X, tr_Y)
            #predico i valori per le clip degli eeg di test
            y_pred = pred.predict(te_X)
            #ricostruisco la predizione del singolo eeg facendo la media delle età nel tempo
            y_pred = [y_pred[eeg_limits_idx[i]:eeg_limits_idx[i+1]].mean() for i in range(len(eeg_limits_idx)-1)]
            #calcolo le metriche sul risultato aggregato dell'eeg
            par_mae.append(mean_absolute_error(y_pred=y_pred, y_true=te_Y))
            par_r2.append(r2_score(y_pred=y_pred, y_true=te_Y))
        partial_res.append(((name+'_MAE', np.mean(par_mae)),(name+'_r2', np.mean(par_r2))))
    results[model_name] = partial_res

rows = []
for model_name, metrics_list in results.items():
    for mae_tuple, r2_tuple in metrics_list:
        rows.append({
            "Embedding": model_name,
            "Predictor": mae_tuple[0].replace("_MAE",""),
            "MAE": mae_tuple[1],
            "R2": r2_tuple[1]
        })

df = pd.DataFrame(rows)
path = os.path.join(out_dir,'age_results.csv')
df.to_csv(path, index = False)

"""
Nella successiva parte di codice calcolo il PCC tra le variabli latenti e le età.
Prendo come valore delle variabili latenti per un soggetto la mediana dei valori latenti lungo la dimensione temporale 
"""
for model_name, model_file, param in zip(model_names, model_paths, params):
    model = load_models(model=model_name, save_files=model_file, params=param)
    _,_,latent = get_orig_rec_latent(raws, model)
    #latent ha forma [N, ch_num, clip_num, 50]
    #calcolo la mediana lungo l'asse temporale per ogni soggetto e per ogni canale
    latent = np.median(latent, axis=2)
    #per ogni soggetto calcolo il pcc su ogni dimensione latente lungo ogni canale
    _, ch_num, latent_dim = latent.shape
    pcc = np.zeros((ch_num, latent_dim))

    for ch in range(ch_num):
        for l_var in range(latent_dim):
            pcc[ch,l_var] = pearsonr(latent[:, ch, l_var], ages)[0]
    
    df = pd.DataFrame(pcc, index=['FP1', 'FP2', 'F3', 'F4', 'FZ', 'F7', 'F8', 'P3', 'P4', 'PZ', 'C3', 'C4', 'CZ', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2'], columns=[f"latent_{d}" for d in range(pcc.shape[1])])
    df.to_csv(model_name+"_pearson_latent.csv", index=True)
