import torch
import numpy as np
import pickle
from model import VAEEG
from sklearn.decomposition import  PCA, FastICA, KernelPCA
import pywt 
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
from utils import stride_data

STANDARD_1020 =  ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'CZ', 'C3', 'C4', 'PZ', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']

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
            #prendo le freq corrispondenti al'onda scelta
            pick_index = np.logical_and(freq<=v[1], freq>v[0])
            #calcolo l'errore medio
            mean_mae = np.abs((orig_phase[pick_index] - rec_phase[pick_index])).mean()
            res[k] = [mean_mae]
        return pd.DataFrame(res)

    def work(self, save_path, orig, rec):
        res = []
        for o,r in zip(orig, rec):
            #per ogni coppia originale ricostruzione ottengo un dataframe con le differenze per secondo 
            #tra le fasi per ogni onda e le salvo in un pd.dataframe
            tmp_res = self.compare_morlet_wavelet_phase(orig=o, rec=r, l_freq=1, h_freq=30, fs=250)
            res.append(tmp_res)
        #poi concateno tutti i dataframe
        res = pd.concat(res, axis=0)
        res.to_csv(save_path, index=False)


class ConComparison():
    def __init__(self, model, save_files, params):
        self.model = self.load_model(model, save_files, params)
    
    def load_model(self, mode, save_files, params=None):
        if mode == 'VAE':
            model = GetLatentVAEEG(*save_files, *params)
        
        if mode == 'PCA':
            with open(save_files,'rb') as f:
                model = pickle.load(f)
        
        if mode == 'KPCA':
            with open(save_files, 'rb'):
                model = pickle.load(f)
        
        if mode == 'FastICA':
            with open(save_files,'rb'):
                model = pickle.load(f)
    
    def complex_data(self, data):
        if not isinstance(data, np.ndarray) or not data.ndim >= 1:
            raise TypeError("data must be a numpy.ndarray with dimension >=1 !")
        
        n_times = data.shape[-1]
        if data.dtype in (np.float32, np.float64):
            n_fft = next_fast_len(n_times)
            analityc_data = hilbert(data, N=n_fft, axis=-1)[..., :n_times]
        else:
            raise ValueError('data.dtype must be float or complex, got %s'
                         % (data.dtype,))
        return analityc_data


    def get_connectivity(self, data, start=None, stop=None, stride=5):
        assert data.shape[0] == len(STANDARD_1020), "input data shape first dim must equal to channels setting."
        data = self.complex_data()
        row_index, col_index = np.triu_indices(19,1)
        paris_len = len(row_index)
        data = stride_data(data, int(250*stride),0) if stride else data
        



    def work(self, save_path, orig, rec):
        
