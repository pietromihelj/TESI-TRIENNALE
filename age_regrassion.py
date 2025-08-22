from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  mean_absolute_error, r2_score
from deploy import load_models, get_orig_rec_latent
import utils as u
import numpy as np
import pandas as pd
import os

#carico i path dei dati da rivedere con il nuovo dataset
paths_list = u.get_path_list(d_path=d_path, f_extensions=['.edf'])
df = pd.read_csv('nchsdb-dataset-0.3.0.csv')

#carico i dati raw e creo l'array delle età. da rivedere con nuovo dataset
raws = []
ages = []
for path in paths_list:
    raw = u.get_raw(path)
    raw.pick_channels(channels)
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

#itero su ogni modello di embedding
for i,model_name, model_file in enumerate(zip(model_names,model_paths):
    #per ognungo creo la stratifiedkfold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    quantiles = pd.qcut(ages, q=5, labels=False)
    #prendo le var latent
    mode=load_models(model, model_file)
    _,_,latent = get_orig_rec_latent(raws, mode)
    latents.append(latent)
    #partial res conterra i risultati per il modello
    partial_res = []
    #per ogni predittore eseguo la k_fold
    for name,pred in predictors:
        par_mae = []
        par_r2 = []
        for idx, (tr_idx, te_idx) in skf.split(latents, quantiles):
            tr_X = latents[tr_idx]
            te_X = latents[te_idx] 
            tr_Y = ages[tr_idx]
            te_Y = ages[te_idx]
            pred.fit(tr_X, tr_Y)
            y_pred = pred.predict(te_X)
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






    

