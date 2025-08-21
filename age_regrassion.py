from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from deploy import load_models, get_orig_rec_latent
import utils as u
import numpy as np
import pandas as pd
import os

paths_list = u.get_path_list(d_path=d_path, f_extensions=['.edf'])
df = pd.read_csv('nchsdb-dataset-0.3.0.csv')

data = []
for path in paths_list:
    raw = u.get_raw(path)
    raw.pick_channels(channels)
    raw = raw.get_data()
    raw = np.array(raw, np.float64)
    raw = raw[:,cuts[0]:cuts[1]]
    file_id = os.path.splitext(os.path.basename(path))[0]
    data.append((raw,int(df.loc[df['filename_id'] == file_id, 'age_at_sleep_study_days'].iloc[0])))

latents = []
for model, model_file in zip(model_names,model_paths):
    mode=load_models(model, model_file)
    _,_,latent = get_orig_rec_latent(raws, mode)

latents = np.array(latents)



    

