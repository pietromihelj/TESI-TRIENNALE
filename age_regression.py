from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  mean_absolute_error, r2_score
from deploy import load_models, get_orig_rec_latent
from scipy.stats import pearsonr
import utils as u
import numpy as np
import pandas as pd
import os

d_path = []
channels = []
cuts = []
model_names = []
model_paths = []
params = []

#carico i path dei dati da rivedere con il nuovo dataset
paths_list = u.get_path_list(d_path=d_path, f_extensions=['.edf'])
df = pd.read_csv('nchsdb-dataset-0.3.0.csv')

#carico i dati raw e creo l'array delle età. da rivedere con nuovo dataset
raws = []
ages = []
for path in paths_list:
    raw = u.get_raw(path)
    raw.pick_channels(channels)
    channels = raw.info['ch_names']
    raw = raw.get_data()
    raw = np.array(raw, np.float64)
    raw = raw[:,cuts[0]:cuts[1]]
    file_id = os.path.splitext(os.path.basename(path))[0]
    raws.append(raw)
    ages.append(int(df.loc[df['filename_id'] == file_id, 'age_at_sleep_study_days'].iloc[0]/365))
ages = np.array(ages)

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
    #per ognungo creo la stratifiedkfold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    quantiles = pd.qcut(ages, q=5, labels=False)
    #prendo le var latent
    model=load_models(model_name, model_file, param)
    _,_,latent = get_orig_rec_latent(raws, model)
    latent = u.stride_data(latent, n_per_seg=50, n_overlap = 0)
    #latent ha forma [N, ch_num, clip_num, 50]
    latent = latent.reshape(latent.shape[0], latent.shape[2], -1)
    #partial res conterra i risultati per il modello
    partial_res = []
    #per ogni predittore eseguo la k_fold
    for name,pred in predictors:
        par_mae = []
        par_r2 = []
        for idx, (tr_idx, te_idx) in skf.split(latent, quantiles):
            tr_X = latent[tr_idx]
            te_X = latent[te_idx] 
            tr_Y = ages[tr_idx]
            te_Y = ages[te_idx]
            pred.fit(tr_X, tr_Y)
            y_pred = pred.predict(te_X)
            y_pred = np.mean(y_pred)
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
df.to_csv('age_results.csv', index = False)

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
    
    df = pd.DataFrame(pcc, index=channels, columns=[f"latent_{d}" for d in range(pcc.shape[1])])
    df.to_csv(model_name+"_pearson_latent.csv", index=True)
