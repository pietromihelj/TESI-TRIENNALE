import torch
from model import VAEEG
import numpy as np
import mne
from utils import stride_data
import pickle
import pandas as pd
import pywt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
from sklearn.decomposition import PCA, FastICA

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
    #calcolo manuale dell'nmsre
    squared_error = (x - y) ** 2
    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)
    mean_x = np.mean(x)
    if mean_x == 0:
        return np.inf
    nrmse = rmse / mean_x
    return nrmse


class DeployVAEEG():
    def __init__(self, paths, params):
        """
        INPUT: Paths dei file di salvataggio dei sottomodelli, parametri
        Carica i file di salvataggio
        Crea i VAEEG in base ai parametri specifici e poi carica i pesi salvati
        Infine sposta il modello su cpu e lo mette in modalità eval
        """
        #carico i file
        files = [torch.load(path) for path in paths]
        #creo i modelli
        self.models = [VAEEG(1,*param) for param in params]
        #carico i pesi
        for model, file in zip(self.models, files):
            model.load_state_dict(file['model'])
        #assicuro che i modelli sia su cpu e li metto in modalità evaluation
        device = 'cpu'
        for model in self.models:
            model.to(device)
            model.eval()

    def preprocess(self, signal, fs=250):
        """
        INPUT: segnale come EEG raw [n_channels, temp_len]
        OUTPUT: torch tensor della forma=[ch_num, clip_num, bands_num, clip_len]
        """
        #inizializzo scala e bande
        scale = 1.0e6
        BANDS =  [("delta", (1.0, 4.0)),
                  ("theta", (4.0, 8.0)),
                  ("alpha", (8.0, 13.0)),
                  ("low_beta", (13, 20)),
                  ("high_beta", (20, 30.0))]
        #estraggo i soli dati come array numpy
        signal = signal.get_data()
        signal = np.array(signal, np.float64)
        signal = signal * scale
        #sistemo la frequenza di samplinf
        if fs != 250:
            ratio = 250/fs
            signal = mne.filter.resample(signal, ratio, verbose=False)
        out = []
        #separo il segnale in 5 bande
        for _, (lf, hf) in BANDS:
            band = mne.filter.filter_data(signal, 250, l_freq=lf, h_freq=hf, filter_length='auto', fir_design='firwin',verbose=False).astype(np.float32) 
            #separo il segnale in clip lunghe 1 secondo
            clips = stride_data(band, 250, 0)
            out.append(clips)
        #ritorno gli array delle bande uniti e riordino per avere [ch_num, clip_num, band_num, clip_len]
        return torch.from_numpy(np.transpose(np.stack(out, axis=0), (1,2,0,3))).float()

    def run(self, inputs):
        """
        INPUT: Un segnale EEG monocanale. in_dim = [clip_num, 5, clip_len]
        OUTPUT: torch tensor segnale EEG ricostruito [temp_len] e le variabili latenti estratte [clip_num, 50]        
        """
        assert isinstance(inputs,torch.tensor), 'il segnale deve essere un aìtensore torch'
        #per ogni banda passo le clip al modello ed estraggo le ricostruzioni e le variabili latenti
        rec_signal = []
        latent_signal = []
        #ogni banda ha forma [clip_num, 1, clip_len]
        for i in range(inputs.shape[1]):
            rec, z = self.models[i](inputs[:,i,:])
            rec_signal.append(rec)
            latent_signal.append(z)
        return torch.stack(rec_signal, dim = 1).sum(dim=1).flatten(), torch.cat(latent_signal, dim=1)
        


class DeployBaseline():
    def __init__(self, path):
        """
        INPUT: Path file di salvataggio modello
        Carica il modello
        """
        with open(path,'rb') as f:
            self.model = pickle.load(f)
        print('Modello caricato')

    def preprocess(self, signal, fs=250):
        """
        INPUT: segnale EEG raw [ch_num, temp_len]
        OUTPUT: numpy array segnale segmentato in clip da 1 secondo [ch_num, clip_num, clip_len]
        """
        #estraggo i dati e li scalo
        scale = 1.0e6
        signal = signal.get_data()
        signal = np.array(signal, np.float64)
        signal = signal * scale
        #sistemo la frequenza
        if fs != 250:
            ratio = 250/fs
            signal = mne.filter.resample(signal, ratio, verbose=False)
        #separo in clip
        clips = stride_data(signal, 250,0)
        return clips

    def run(self, inputs):
        """
        INPUT: segnale EEG monocanale [clip_num, clip_len] 
        OUTPUT: numpy array segnale ricostruito [temp_len] e le variabili latenti estratte [clip_num,50]
        """
        #ricavo le latenti per ogni clip
        z = self.model.transform(inputs)
        #prendo le ricostruzioni per ogni clip
        rec = self.model.inverse_transform(z)
        return rec.flatten(), z

def get_orig_rec_latent(raws, model):
    """
    INPUT: lista di segnali raw EEG
    OUTPUT: numpy array di originali = [N, ch_num, temp_len], ricostruzioni = [N, ch_num, temp_len], latenti dim [N, ch_num, clip_num, 50]
    """
    assert isinstance(raws, list), 'input deve essere una lista'
    if isinstance(model, DeployVAEEG):
        origs = []
        #origin diventa una lista di tensori di forma [ch_num, clip_num, bands_num, clip_len]
        for raw in raws:
            origs.append(model.preprocess(raw))
        
        rec = []
        latent = []
        for ori in origs:
            ch_rec = []
            ch_latent = []
            #ogni canale ha forma [clip_num, bands_num, clip_len]
            for ch in ori:
                c_r, c_l = model.run(ch)
                ch_rec.append(c_r.detach().cpu().numpy())
                ch_latent.append(c_l.detach().cpu().numpy())
            rec.append(np.stack(ch_rec))
            latent.append(np.stack(ch_latent))
        
        for o in origs:
            o.sum(dim=2)
            o.reshape((o.shape[0], -1)).detach().cpu().numpy()

        return np.stack(origs), rec, latent 
    #[ch_num, clip_num, bands_num, clip_len]
    if isinstance(model, (PCA, FastICA)):
        origis = []
        for raw in raws:
            origis.append(model.preprocess())
        
        rec = []
        latent = []
        for ori in origis:
            ch_rec = []
            ch_latent = []
            for ch in ori:
                c_r, c_l = model.run(ch)
                ch_rec.append(c_r)
                ch_latent.append(c_l)
            rec.append(np.stack(ch_rec))
            latent.append(np.stack(ch_latent))
        return np.stack(origis, axis=0).reshape(origis.shape[0], origis.shape[1], -1), np.array(rec), np.array(latent)


    

class PhaseComparison():
    def __init__(self):
        self.BANDS =  [("delta", (1.0, 4.0)),
                  ("theta", (4.0, 8.0)),
                  ("alpha", (8.0, 13.0)),
                  ("low_beta", (13, 20)),
                  ("high_beta", (20, 30.0))]
    
    def compare_mo_wa_ps(self, orig, rec, l_freq=1, h_freq=30, fs=250):
        """
        INPUT: segnale e la sua ricostruzione monocanale [temp_len], come array numpy
        OUTPUT: dizionario con l'errore medio per banda della differenza di fase
        """
        assert isinstance(orig, np.array) and isinstance(rec, np.array), 'input deve essere numpy array'
        #creo la lista delle scale per trovare le frequenze
        scales = np.arange(fs/2/h_freq, fs/2/l_freq, 1)
        #calcolo le trasformate wavelet morlet
        coeff_orig, freq = pywt.cwt(orig, wavelet='cmor', scales=scales, sampling_period=1/fs)
        coeff_rec, freq = pywt.cwt(rec, wavelet='cmor', scales=scales, sampling_period=1/fs)
        #calcolo le fasi
        orig_phase, rec_phase = np.angle(coeff_orig), np.angle(coeff_rec)

        #calcolo la media della differenza di fase per banda
        res = {}
        for k,v in self.BANDS.items():
            #prendo le frequenze corrispondenti alla banda
            pick_index = np.logical_and(freq<=v[1], freq>v[0])
            #calcolo l'errore medio
            res[k] = [np.abs((orig_phase[pick_index])-rec_phase[pick_index])]
        return res
    
    def work(self, raws, model):
        """
        INPUT: lista di segnali raw
        OUTPUT: Dataframe contenente l'errore medio di fase per banda
        """
        orig, rec, _ = get_orig_rec_latent(raws, model) 
        res = []
        #per ogni coppia di valori calcolo la media per canale
        for o,r in zip(orig, rec):
            ch_res = []
            for ch_o, ch_r in zip(o,r):
                #calcolo l'errore di phase medio per banda
                temp_res = self.compare_mo_wa_ps(ch_o.reshape(-1),ch_r.reshape(-1))
                ch_res.append(list(temp_res.values()))
            #calcolo poi l'errore medio tra i canali per banda
            ch_res = np.stack(ch_res, axis=1)
            ch_res = np.mean(ch_res,axis=1)
        res.append(ch_res)
        #calcolo poi la media tra le coppie per banda
        return pd.DataFrame(dict(zip(list(self.BANDS.keys()),np.mean(np.stack(res, axis=1), axis=1).tolist())))


class ConComparison():
    def __init__(self, model, save_files, params=None):
        """
        INPUT: Tipo di modello, file dove è salvato e parametri
        Carica la classe di deploy del modello da usare
        """
        if model == 'VAEEG':
            self.model = DeployVAEEG(*save_files, *params)
        elif model in ['PCA','KernelPCA','FastICA']:
            model=DeployBaseline(save_files)
        else:
            raise Exception('Il modello specificato non è supportato')
        
    def complex_data(self, data):
        """
        INPUT: Segnale EEG [ch_num, temp_len]
        OUTPUT: Segnale EEG complesso
        """
        if not isinstance(data, np.ndarray) or not data.ndim >= 1:
            raise TypeError("data must be a numpy.ndarray with dimension >=1 !")
        #prendo la dimensione temporale
        n_times = data.shape[-1]
        #calcolo la trasformata di hilbert
        if data.dtype in (np.float32, np.float64):
            n_fft = next_fast_len(n_times)
            analityc_data = hilbert(data, N=n_fft, axis=-1)[..., :n_times]
        else:
            raise ValueError('data.dtype must be float or complex, got %s'
                         % (data.dtype,))
        return analityc_data

    def pcc_con(self, data):
        """
        INPUT: segnale multicanale [ch_num, temp_len]
        OUTPUT: matrice di correlazione tra i canali
        """
        assert isinstance(data, np.array), 'input deve essere numpy array'
        #calcolo la matrice di correlazione
        data = data - data.mean(axis=-1, keepdims=True)
        div_on = np.matmul(data, data.T)
        #normalizzo
        tmp = (data**2).sum(axis=-1, keepdims=True)
        div_down = np.sqrt(np.matmul(tmp,tmp.T))
        return div_on/div_down

    def pvl_con(self, data_a, data_b):
        """
        INPUT: 2 segnali monocanale complessi [temp_len]
        OUTPUT: phase lock value tra i 2 canali
        """
        assert data_a.dtype in (np.complex64, np.complex128) and data_b.dtype in (np.complex64, np.complex128),  'data type must be complex , got %s %s'%(data_a.dtype, data_b.dtype)
        #calcolo la fase
        data_a = np.arctan(data_a.imag / data_a.real)
        data_b = np.arctan(data_b.imag / data_b.real)
        #calcolo la differenza di fase per campione
        t = np.exp(np.complex(0,1)*(data_a-data_b))
        #calcolo il pvl 
        t_len = t.shape[-1]
        t = np.abs(np.sum(t,axis=-1))/t_len
        return t        

    def get_con(self, orig, rec):
        """
        INPUT: 2 segnali interi contenenti 19 canali
        OUTPUT: le 4 matrici di connettività per
        """
        assert orig.shape[0] == len(STANDARD_1020) and rec.shape[0] == len(STANDARD_1020), 'i segnali devono avere i 19 canali dello standard 10-20'
        ch_num = orig.shape[0]
        #calcolo le matrici di pcc
        orig_pcc = self.pcc_con(orig)
        rec_pcc = self.pcc_con(rec)

        #inizializzo le matrici di pvl
        orig_pvl = np.zeros((ch_num,ch_num))
        rec_pvl = np.zeros((ch_num,ch_num))

        #per ogni coppia di canali calcolo il pvl e poi lo specchio nella matrice
        for i in range(ch_num):
            for j in range(i+1, ):
                orig_pvl[i, j] = self.pvl_con(orig[i], orig[j])
                orig_pvl[j, i] = orig_pvl[i, j]  
                rec_pvl[i,j] = self.pvl_con(rec[i], rec[j])
                rec_pvl[j,i] = rec_pvl[i,j]
        return orig_pcc, orig_pvl, rec_pcc, rec_pvl
    
    def work(self, raws, model):
        """
        INPUT: Batch di segnali raw
        OUTPUT: Matrici medie della connettività tra i segnali
        """
        #calcolo origin e rec
        orig, rec, _ = get_orig_rec_latent(raws, model)

        orig_pvl = []
        orig_pcc = []
        rec_pvl = []
        rec_pcc = []
        for o, r in zip (orig, rec):
            o_pcc, o_pvl, r_pcc, r_pvl = self.get_con(o,r)
            orig_pcc.append(o_pcc)
            orig_pvl.append(o_pvl)
            rec_pcc.append(r_pcc)
            rec_pvl.append(r_pvl)
        orig_pcc = np.mean(np.stack(orig_pcc, axis=0), axis=0)
        orig_pvl = np.mean(np.stack(orig_pvl, axis=0), axis=0)
        rec_pcc = np.mean(np.stack(rec_pcc, axis=0), axis=0)
        rec_pvl = np.mean(np.stack(rec_pvl, axis=0), axis=0)
        
        return orig_pcc, orig_pvl, rec_pcc, rec_pvl


def evaluate(data_dir, model_files, out_dir):
    """
    INPUT: dove trovare i dati e dove trovare i modelli
    OUTPUT: Il CCP medio orig/rec, l'NMRSE medio orig/rec, il dataframe delle differenze di fase, le matrici di connettività.
    """

    

