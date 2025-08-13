import torch
import numpy as np
import pickle
from model import VAEEG
from sklearn.decomposition import  PCA, FastICA, KernelPCA
import pywt 
import pandas as pd



def pearson_index(x, y, dim=-1):
        #calcolo manuale del coeff di correlazione di pearson lungo una specifica dimensione
        xy = x * y
        xx = x * x
        yy = y * y

        mx = x.mean(dim)
        my = y.mean(dim)
        mxy = xy.mean(dim)
        mxx = xx.mean(dim)
        myy = yy.mean(dim)

        varx = mxx - mx ** 2
        vary = myy - my ** 2

        if vary==0:
            print('vary = 0')
            return np.inf

        r = (mxy - mx * my) / torch.sqrt(varx*vary)
        return r


def NRMSE(x,y):
    squared_error = (x - y) ** 2
    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)
    mean_x = np.mean(x)
    if mean_x == 0:
        return np.inf
    nrmse = rmse / mean_x
    return nrmse


class GetLatentVAEEG():
    
    def __init__(self, am_path, tm_path, dm_path, lbm_path, hbm_path, z_dims, slopes, lstms):
        am = torch.load(am_path)
        tm = torch.load(tm_path)
        dm = torch.load(dm_path)
        lbm = torch.load(lbm_path)
        hbm = torch.load(hbm_path)
        self.models = [VAEEG(1, *params)  for _,params in enumerate(zip(z_dims,slopes,lstms))]
        for model, saves in zip(self.models,[am,tm,dm,lbm,hbm]):
            model.load_state_dict(saves['model'])
        self.device = 'cpu'
        for m in self.models:
            m.to(self.device)
            m.eval()

    def run(self,inputs):
        if not torch.is_tensor(inputs):
            raise Exception('I dati devono essere un tensore torch ma sono: ', type(inputs))
        x_recs = []
        zs = []
        for i in inputs:
            res = [model(i) for model in self.models]
            x_rec = [x_r for _,_,x_r,_ in res]
            z = [z_ for _,_,_,z_ in res]

            x_recs.append(x_rec)
            zs.append(z)
        return x_recs, zs


class GetLatentBaseline():
    def __init__(self, file_name):
        with open(file_name, 'rb') as f:
            self.model = pickle.load(f)
    
    def run(self, inputs):
        if not isinstance(inputs, np.ndarray):
             raise Exception('I dati devono essere un array numpy ma sono: ', type(inputs))
        
        zs = self.model.transform(inputs)
        x_recon = self.model.inverse_transform(zs)

        return zs, x_recon
    

class PhaseComparison():
    def __init__(self):
        self.BANDS = [("delta", (1.0, 4.0)),
                  ("theta", (4.0, 8.0)),
                  ("alpha", (8.0, 13.0)),
                  ("low_beta", (13, 20)),
                  ("high_beta", (20, 30.0))]
    
    def compare_morlet_wavelet_phase(self, orig, rec, l_freq, h_freq, fs=250):
        #creo la lista di scale
        scales=np.arange(fs/2/h_freq, fs/2/l_freq, 1)
        #alcolo le trasformate wavelet morlet
        coeff_orig, freq = pywt.cwt(orig, wavelet='cmor', scales=scales, sampling_period=1/fs)
        coeff_rec, freq = pywt.cwt(rec, wavelet='cmor', scales=scales, sampling_period=1/fs)
        #calcolo la fase dei segnali
        orig_phase, rec_phase = np.angle(coeff_orig), np.angle(coeff_rec)

        res = {}
        for k,v in self.BANDS.items():
            pick_index = np.logical_and(freq<=v[1], freq>v[0])
            mean_mae = np.abs((orig_phase[pick_index] - rec_phase[pick_index])).mean()
            res[k] = [mean_mae]
        return pd.DataFrame(res)

    def work(self, model, save_path, orig, rec):
        res = []
        for o,r in zip(orig, rec):
            tmp_res = self.compare_morlet_wavelet_phase(orig=o, rec=r, l_freq=1, h_freq=30, fs=250)
            res.append(tmp_res)
        res = pd.concat(res, axis=0)
        res.to_csv(save_path, index=False)


class ConComparison():
