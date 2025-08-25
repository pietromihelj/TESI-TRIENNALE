import numpy as np
import pickle
import torch
import mne
from utils import stride_data
from sklearn.decomposition import PCA, FastICA
from model import VAEEG
import portion as P
from utils import find_artefacts_2d, merge_continuous_artifacts

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

    def preprocess(self, signal, amp_th = 400, m_len=1.0, drop = 60, fs=250):
        """
        INPUT: segnale come EEG numpy [n_channels, temp_len]
        OUTPUT: torch tensor della forma=[ch_num, clip_num, bands_num, clip_len]
        """
        #inizializzo scala e bande
        scale = 1.0e6
        BANDS =  [("delta", (1.0, 4.0)),
                  ("theta", (4.0, 8.0)),
                  ("alpha", (8.0, 13.0)),
                  ("low_beta", (13, 20)),
                  ("high_beta", (20, 30.0))]

        signal = signal * scale
        #sistemo la frequenza di samplinf
        if fs != 250:
            ratio = 250/fs
            signal = mne.filter.resample(signal, ratio, verbose=False)
        
        mth = int(m_len*fs)
        start = int(drop*fs)
        end = signal.shape[1] - int(drop*fs)
        whole_r = P.closed(start, end)
        flag = np.abs(signal) > amp_th
        art = find_artefacts_2d(flag)
        merged_art = [merge_continuous_artifacts(s, mth) for s in art]
        c_clean = [whole_r - s for s in merged_art]
        common_intervals = c_clean[0]
        for ch_intervals in c_clean[1:]:
            common_intervals = common_intervals & ch_intervals
        valid_idx = np.concatenate([np.arange(interval.lower, interval.upper + 1) for interval in common_intervals])
        clean_signal = signal[:,valid_idx]
        out = []
        #separo il segnale in 5 bande
        for _, (lf, hf) in BANDS:
            band = mne.filter.filter_data(clean_signal, 250, l_freq=lf, h_freq=hf, filter_length='auto', fir_design='firwin',verbose=False).astype(np.float32) 
            #separo il segnale in clip lunghe 1 secondo
            clips = stride_data(band, 250, 0)
            out.append(clips)
        #ritorno gli array delle bande uniti e riordino per avere [ch_num, clip_num, band_num, clip_len]
        return torch.from_numpy(np.transpose(np.stack(out, axis=0), (1,2,0,3))).float()

    def run(self, inputs):
        """
        INPUT: Un segnale EEG monocanale. in_dim = [clip_num, 5, clip_len]
        OUTPUT: torch tensor segnale EEG ricostruito [band_num, temp_len] e le variabili latenti estratte [clip_num*50]        
        """
        assert isinstance(inputs,torch.tensor), 'il segnale deve essere un aìtensore torch'
        #per ogni banda passo le clip al modello ed estraggo le ricostruzioni e le variabili latenti
        inputs = inputs.to('cpu')
        rec_signal = []
        latent_signal = []
        #ogni banda ha forma [clip_num, 1, clip_len]
        for i in range(inputs.shape[1]):
            rec, z = self.models[i](inputs[:,i:i+1,:])
            rec_signal.append(rec)
            latent_signal.append(z)
        return rec_signal.transpose(0,1).flatten(1), torch.cat(latent_signal, dim=1)

class DeployBaseline():
    def __init__(self, path):
        """
        INPUT: Path file di salvataggio modello
        Carica il modello
        """
        with open(path,'rb') as f:
            self.model = pickle.load(f)
        print('Modello caricato: ', type(self.model) )

    def preprocess(self, signal, amp_th = 400, m_len=1.0, drop = 60 ,fs=250):
        """
        INPUT: segnale EEG numpy [ch_num, temp_len]
        OUTPUT: numpy array segnale segmentato in clip da 1 secondo [ch_num, clip_num, clip_len]
        """
        #estraggo i dati e li scalo
        scale = 1.0e6
        #signal = signal * scale
        #sistemo la frequenza
        if fs != 250:
            ratio = 250/fs
            signal = mne.filter.resample(signal, ratio, verbose=False)
        mth = int(m_len*fs)
        start = int(drop*fs)
        end = signal.shape[1] - int(drop*fs)
        whole_r = P.closed(start, end)
        flag = np.abs(signal) > amp_th
        art = find_artefacts_2d(flag)
        merged_art = [merge_continuous_artifacts(s, mth) for s in art]
        c_clean = [whole_r - s for s in merged_art]
        common_intervals = c_clean[0]
        for ch_intervals in c_clean[1:]:
            common_intervals = common_intervals & ch_intervals
        valid_idx = np.concatenate([np.arange(interval.lower, interval.upper + 1) for interval in common_intervals])
        clean_signal = signal[:,valid_idx]
        #separo in clip
        clips = stride_data(clean_signal, 250,0)
        return clips

    def run(self, inputs):
        """
        INPUT: segnale EEG monocanale [clip_num, clip_len] 
        OUTPUT: numpy array segnale ricostruito [temp_len] e le variabili latenti estratte [clip_num*50]
        """
        #ricavo le latenti per ogni clip
        z = self.model.transform(inputs)
        #prendo le ricostruzioni per ogni clip
        rec = self.model.inverse_transform(z)
        return rec.flatten(), z

def get_orig_rec_latent(raw, model):
    """
    INPUT: lista di segnali EEG numpy
    OUTPUT: numpy array originale = [ch_num, band_num, temp_len], ricostruzione = [ch_num, band_num, temp_len], latente dim [ch_num, clip_num ,50]
    """
    assert isinstance(raw, np.ndarray), 'input deve essere una array numpy'
    if isinstance(model, DeployVAEEG):
        orig = model.preprocess(signal=raw)
        #origin diventa una lista di forma [ch_num, clip_num, bands_num, clip_len]
        ch_rec = []
        ch_latent = []
        #ogni canale ha forma [clip_num, bands_num, clip_len]
        for ch in orig:
            c_r, c_l = model.run(ch)
            ch_rec.append(c_r.detach().cpu().numpy())
            ch_latent.append(c_l.detach().cpu().numpy())
            #ogni cr,cl ha forma [bands_num, temp_len]
        rec = np.stack(ch_rec, axis=0)
        latent= np.stack(ch_latent, axis=0)
        orig=orig.transpose(1,2).flatten(2)
        orig=orig.detach().cpu().numpy()
        #ottengo una array numpy di forma [ch_num, band_num, temp_len]
        return np.array(orig), np.array(rec), np.array(latent) 

    if isinstance(model, DeployBaseline):
        orig = model.preprocess(signal=raw)
            #origis [N, ch_num, clup_num, clip_len]
        ch_rec = []
        ch_latent = []
        #ori [ch_num, clip_num*clip_len]
        for ch in orig:
            #ch [clip_num, clip_len]
            c_r, c_l = model.run(ch)
            #cr,cl [temp_len] = [clip_num*50]
            ch_rec.append(c_r)
            ch_latent.append(c_l)
        rec= np.stack(ch_rec, axis=0)
        latent = (np.stack(ch_latent, axis=0))
        return np.expand_dims(orig.reshape(orig.shape[0],-1),axis=1), np.expand_dims(np.array(rec), axis=1), np.array(latent)
    
def load_models(model, save_files, params=None):
    if model == 'VAEEG':
        model = DeployVAEEG(*save_files, *params)
    elif model in ['PCA','KernelPCA','FastICA']:
        model=DeployBaseline(save_files)
    else:
        raise Exception('Il modello specificato non è supportato')
    return model