import torch
from model import VAEEG
import numpy as np
import mne
from utils import stride_data
import pickle
import pandas as pd
import pywt

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
        ipotesi: segnale raw
        INPUT: segnale monocanale di eeg come numpy [n_channels, temp_len]
        OUTPUT: array numpy della forma=[ch_num, clip_num, bands_num, clip_len]
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
        return np.transpose(np.stack(out, axis=0), (1,2,0,3))

    def run(self, inputs):
        """
        INPUT: Un segnale EEG monocanale. in_dim = [clip_num, 5, clip_len]
        OUTPUT: Il segnale EEG ricostruito [clip_num, clip_len] e le variabili latenti estratte [clip_num, 50]        
        """
        #per ogni banda passo le clip al modello ed estraggo le ricostruzioni e le variabili latenti
        rec_signal = []
        latent_signal = []
        #ogni banda ha forma [clip_num, 1, clip_len]
        for i in range(inputs.shape[1]):
            rec, z = self.models[i](inputs[:,i,:])
            rec_signal.append(rec)
            latent_signal.append(z)
        return torch.stack(rec_signal, dim = 1).sum(axis=1), torch.cat(latent_signal, dim=1)
        
        

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
        OUTPUT: segnale segmentato in clip da 1 secondo [ch_num, clip_num, clip_len]
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
        INPUT: lista di clip monocanele [N,250] 
        OUTPUT: Il segnale ricostruito e le variabili latenti estratte 
        """
        #ricavo le latenti per ogni clip
        z = self.model.transform(inputs)
        #prendo le ricostruzioni per ogni clip
        rec = self.model.inverse_transform(z)
        return rec, z

def get_rec_latent():
    """
    INPUT: lista di segnali raw EEG
    OUTPUT: Lista di ricostruzioni, lista di latenti
    """
    pass

class PhaseComparison():
    def __init__(self):
        self.BANDS =  [("delta", (1.0, 4.0)),
                  ("theta", (4.0, 8.0)),
                  ("alpha", (8.0, 13.0)),
                  ("low_beta", (13, 20)),
                  ("high_beta", (20, 30.0))]
    
    def compare_mo_wa_ps(self, orig, rec, l_freq=1, h_freq=30, fs=250):
        """
        INPUT: segnale e la sua ricostruzione monocanale [temp_len,]
        OUTPUT: dataframe pandas con l'errore medio per banda della differenza di fase
        """
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
    
    def work(self, orig, rec):
        """
        INPUT: lista di segnali raw
        OUTPUT: Dizionario contenente l'errore medio di fase per banda
        """
        res = []
        #per ogni coppia di valori calcolo la media per canale
        for o,r in zip(orig, rec):
            ch_res = []
            for ch_o, ch_r in zip(o,r):
                #calcolo l'errore di phase medio per banda
                temp_res = self.compare_mo_wa_ps(ch_o,ch_r)
                ch_res.append(list(temp_res.values()))
            #calcolo poi l'errore medio tra i canali per banda
            ch_res = np.stack(ch_res, axis=1)
            ch_res = np.mean(ch_res,axis=1)
        res.append(ch_res)
        #calcolo poi la media tra le coppie per banda
        return dict(zip(list(self.BANDS.keys()),np.mean(np.stack(res, axis=1), axis=1).tolist()))


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

    def pcc_con(self, orig_ch, rec_ch):
        """
        INPUT: 2 segnali monocanale, originale e ricostruito
        OUTPUT: valore del coefficiente di pearson tra i 2 canali
        """
        pass

    def pvl_con(self, orig_ch, rec_ch):
        """
        INPUT: 2 segnali monocanale, originale e ricostruito
        OUTPUT: phase lock value tra i 2 canali
        """
        pass

    def get_con(self, orig, rec):
        """
        INPUT: 2 segnali interi contenenti 19 canali e tutte e 5 le bande

        La funzione si occupa di iterare su tutte le coppie di canali e chiamare le funzioni di 
        calcolo della connettività

        OUTPUT: 2 liste contenenti i valori delle connettività tra i 2 segnali per ogni coppia di canali
        """
        pass

    def work():
        """
        INPUT: Batch di segnali raw

        deve essere eseguito il preprocessing per ogni segnale, poi ottenuta la ricostruzione del segnale
        ed infine passato ogni segnale alla funzione di calcolo della connettivita

        OUTPUT: Dataframe contenente le 2 liste con il valore delle connettività per ogni segnale
        """
