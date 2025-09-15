from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  mean_absolute_error, r2_score
from scipy.stats import pearsonr
from utils import get_path_list, stride_data
import numpy as np
import pandas as pd
import os
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning



with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
w_len = 0

parser = argparse.ArgumentParser(description='Age Regression')
parser.add_argument('--out_dir', type=str, required=True, help='out_dir')

opts = parser.parse_args()

out_dir = opts.out_dir

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
"""
Nella successiva parte di codice carico i dati e creo 2 liste contenenti la coppia eeg eta soggetto
"""
path = get_path_list("C:/Users/Pietro/Desktop/age_eeg/rbf_10", f_extensions=['.npy'], sub_d=True)
age_df = pd.read_csv("D:/nmt_scalp_age_dataset/Labels.csv")
ages = []
latents  = []
i = 0
total_mean = ()
for j, p in enumerate(path):
    lat = np.load(p,allow_pickle=True)
    if len(lat.shape) == 0:
        i = i+1
        continue
    latents.append(np.hstack([lat.reshape(-1,50), np.full((lat.reshape(-1,50).shape[0],1),j-i)]))
    code = os.path.splitext(os.path.basename(p))[0]+'.edf'
    ages.append(age_df.loc[age_df['recordname'] == code, 'age'].iloc[0])
#ottengo un unico array con tutte le clip. in questo modo diventano il dato singolo
latents = np.concatenate(latents, axis=0)

print('AGES: ', len(ages))
print('RAWS: ', latents.shape)


#inizzializzo i predittori di test
predictors = [
    ("LinearRegression", LinearRegression()),
    ("Lasso", Lasso(alpha=1.0)),
    ("Ridge", Ridge(alpha=1.0)),
]

results = {}
print(f'Regressione con variabili del modello: VAEEG')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
for lat in latents:
    try:
        a=ages[int(lat[-1])]
    except Exception:
        print(lat[-1])
        print(ages[int(lat[-1])])
clip_ages = np.array([ages[int(lat[-1])] for lat in latents])
quantiles = pd.qcut(clip_ages, q=5, labels=False)
partial_res = []

for name, pred in predictors:
    print('Training predittore: ', name)
    par_mae = []
    par_r2 = []
    for l,(tr_idx, te_idx) in enumerate(skf.split(latents, quantiles)):
        print('training_split: ', l)
        #estraggo le clip di training e rimuovo il sid
        tr_X = latents[tr_idx,:-1]
        #creo le labels di training
        tr_Y = clip_ages[tr_idx]
        #estraggo le clip di test allo stesso modo di quelle di train
        te_X = latents[te_idx,:-1]
        #creo le label reali alllo stesso modo
        te_Y = clip_ages[te_idx]
        te_sids = np.array([int(eeg[-1]) for eeg in latents[te_idx]])
        #alleno il modello
        pred.fit(tr_X, tr_Y)
        for warn in w:
            if issubclass(warn.category, ConvergenceWarning) and len(w) > w_len:
                print("⚠️ Warning catturato: non convergenza modello: ", name)
            w_len = w_len+1
            break
        y_pred = pred.predict(te_X)
        #ricostruisco ora la predizione per l'eeg aggregato usando il sid
        y_pred = pd.DataFrame({'sid': te_sids, 'Y_pred': y_pred})
        y_pred = y_pred.groupby('sid')['Y_pred'].mean().values
        #ricostruisco anche le vere labels
        te_Y = pd.DataFrame({'sid': te_sids, 'labels': te_Y})
        y_true = te_Y.groupby('sid')['labels'].mean().values
        par_mae.append(mean_absolute_error(y_pred=y_pred, y_true=y_true))
        par_r2.append(r2_score(y_pred=y_pred, y_true=y_true))
    partial_res.append(((name+'_MAE', np.mean(par_mae)),(name+'_R2', np.mean(par_r2))))
results['KernelPCA'] = partial_res

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
"""
path = get_path_list("C:/Users/Pietro/Desktop/age_eeg/rbf_0.1", f_extensions=['.npy'], sub_d=True)
pccs = []
lat_value = []
for i,p in enumerate(path):
    #calcolo la mediana lungo l'asse temporale per ogni soggetto e per ogni canale
    lat = np.load(p,allow_pickle=True)
    lat_value.append(np.median(lat, axis=1))
    #per ogni soggetto calcolo il pcc su ogni dimensione latente lungo ogni canale
lat_value = np.array(lat_value)
_,ch_num, latent_dim = lat_value.shape
pcc = np.zeros((ch_num, latent_dim))
for ch in range(ch_num):
    for l_var in range(latent_dim):
        pcc[ch,l_var] = pearsonr(lat_value[:,ch, l_var], ages)[0]
    pccs.append(pcc)
pccs = np.array(pccs).mean(axis=0)
df = pd.DataFrame(pcc, index=['FP1', 'FP2', 'F3', 'F4', 'FZ', 'F7', 'F8', 'P3', 'P4', 'PZ', 'C3', 'C4', 'CZ', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2'], columns=[f"latent_{d}" for d in range(pcc.shape[1])])
df.to_csv(model_name+"_pearson_latent.csv", index=True)
"""