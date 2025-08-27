import torch
import numpy as np
import pandas as pd
import pywt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
import utils
import matplotlib.pyplot as plt
import os
from deploy import get_orig_rec_latent, load_models
from tqdm import tqdm

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

        r = (mxy - mx * my) / np.sqrt(varx*vary)
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

class PhaseComparison():
    def __init__(self):
        self.BANDS = {"delta": (1.0, 4.0),
                      "theta": (4.0, 8.0),
                      "alpha": (8.0, 13.0),
                      "low_beta": (13, 20),
                      "high_beta": (20, 30.0)}
    
    def compare_mo_wa_ps(self, orig, rec, l_freq=1, h_freq=30, fs=250):
        """
        INPUT: segnale e la sua ricostruzione monocanale [temp_len], come array numpy
        OUTPUT: dizionario con l'errore medio per banda della differenza di fase
        """
        assert isinstance(orig, np.ndarray) and isinstance(rec, np.ndarray), 'input deve essere numpy array'
        #creo la lista delle scale per trovare le frequenze e uso una scala logaritmi per lo spacing
        freqs = np.logspace(np.log10(l_freq), np.log10(h_freq), num=60)
        fc = pywt.central_frequency('cmor1.5-1.0')
        scales = fc / (freqs * 1/fs)
        #calcolo le trasformate wavelet morlet
        coeff_orig, freq = pywt.cwt(orig, wavelet='cmor1.5-1.0', scales=scales, sampling_period=1/fs)
        coeff_rec, freq = pywt.cwt(rec, wavelet='cmor1.5-1.0', scales=scales, sampling_period=1/fs)
        #calcolo le fasi
        orig_phase, rec_phase = np.angle(coeff_orig), np.angle(coeff_rec)

        #calcolo la media della differenza di fase per banda
        res = []
        for k,v in self.BANDS.items():
            #prendo le frequenze corrispondenti alla banda
            pick_index = np.logical_and(freq<=v[1], freq>v[0])
            #calcolo l'errore medio
            res.append(np.mean(np.abs((orig_phase[pick_index])-rec_phase[pick_index])))
        return res
    
    def work(self, orig, rec):
        """
        INPUT: lista di segnali EEG
        OUTPUT: Dataframe contenente l'errore medio di fase per banda
        """
        #orig,rec hanno forma [ch_num, band_num, temp_len]
        orig = [o.sum(axis=1) for o in orig]
        rec = [r.sum(axis=1) for r in rec]
        res = []
        #per ogni coppia di valori calcolo la media per canale
        print('CHECKPOINT: comparazione di fase')
        for o,r in tqdm(zip(orig, rec)):
            #o,r hanno forma [ch_num, temp_len]
            ch_res = []
            for ch_o, ch_r in zip(o,r):
                #ch_o,ch-r hanno forma [temp_len]
                ch_res.append(self.compare_mo_wa_ps(ch_o.reshape(-1),ch_r.reshape(-1)))
            #calcolo poi l'errore medio tra i canali per banda
            ch_res = np.mean(ch_res,axis=0)
            res.append(ch_res)
        #calcolo poi la media tra le coppie per banda
        return pd.DataFrame({'BANDA':list(self.BANDS.keys()), 'PMAE':np.mean(res, axis=0)}).to_numpy()


class ConComparison():
    def __init__(self):
        """
        INPUT: Tipo di modello, file dove è salvato e parametri
        Carica la classe di deploy del modello da usare
        """
        
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
        assert isinstance(data, np.ndarray), 'input deve essere numpy array'
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
        assert data_a.dtype == complex and data_b.dtype == complex,  'data type must be complex , got %s %s'%(data_a.dtype, data_b.dtype)
        #calcolo la fase
        data_a = np.arctan(data_a.imag / data_a.real)
        data_b = np.arctan(data_b.imag / data_b.real)
        #calcolo la differenza di fase per campione
        t = np.exp(complex(0,1)*(data_a-data_b))
        #calcolo il pvl 
        t_len = t.shape[-1]
        t = np.abs(np.sum(t,axis=-1))/t_len
        return t        

    def get_con(self, orig, rec):
        """
        INPUT: 2 segnali interi contenenti 19 canali monobanda
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
            or_i = self.complex_data(orig[i])
            re_i = self.complex_data(rec[i])
            for j in range(i+1, ):
                or_j = self.complex_data(orig[j])
                re_j = self.complex_data(rec[j])
                orig_pvl[i, j] = self.pvl_con(or_i, or_j)
                orig_pvl[j, i] = orig_pvl[i, j]  
                rec_pvl[i,j] = self.pvl_con(re_i, re_j)
                rec_pvl[j,i] = rec_pvl[i,j]
        return orig_pcc, orig_pvl, rec_pcc, rec_pvl
    
    def work(self, orig, rec):
        """
        INPUT: Batch di segnali EEG
        OUTPUT: Matrici di connettività tra i segnali per banda [N, band_num, ch_num, ch_num]
        """
        #orig,rec hanno forma [N, ch_num, band_num, temp_len]
        orig_pvl = []
        orig_pcc = []
        rec_pvl = []
        rec_pcc = []
        print('CHECKPOINT: comparazione connettività')
        for o,r in tqdm(zip(orig, rec)):
            #o,r hanno forma [ch_num, band_num, temp_len]
            band_o_pcc = []
            band_o_pvl = []
            band_r_pcc = []
            band_r_pvl = []
            for i in range(o.shape[1]):
                o_pcc, o_pvl, r_pcc, r_pvl = self.get_con(o[:,i,:],r[:,i,:])
                band_o_pcc.append(o_pcc)
                band_o_pvl.append(o_pvl)
                band_r_pcc.append(r_pcc)
                band_r_pvl.append(r_pvl)
            orig_pcc.append(band_o_pcc)
            orig_pvl.append(band_o_pvl)
            rec_pcc.append(band_r_pcc)
            rec_pvl.append(band_r_pvl)
        orig_pcc = np.stack(orig_pcc, axis=0)
        orig_pvl = np.stack(orig_pvl, axis=0)
        rec_pcc = np.stack(rec_pcc, axis=0)
        rec_pvl = np.stack(rec_pvl, axis=0)
        
        return orig_pcc, orig_pvl, rec_pcc, rec_pvl

def check_channel_names(raw_obj, verbose):
        ch_map = ["EEG FP1-REF", "EEG FP1-LE", "EEG FP1", "FP1",
        "EEG FP2-REF", "EEG FP2-LE", "EEG FP2", "FP2",
        "EEG F3-REF", "EEG F3-LE", "EEG F3", "F3",
        "EEG F4-REF", "EEG F4-LE", "EEG F4", "F4",
        "EEG FZ-REF", "EEG FZ-LE", "EEG FZ", "FZ",
        "EEG F7-REF", "EEG F7-LE", "EEG F7", "F7",
        "EEG F8-REF", "EEG F8-LE", "EEG F8", "F8",
        "EEG P3-REF", "EEG P3-LE", "EEG P3", "P3",
        "EEG P4-REF", "EEG P4-LE", "EEG P4", "P4",
        "EEG PZ-REF", "EEG PZ-LE", "EEG PZ", "PZ",
        "EEG C3-REF", "EEG C3-LE", "EEG C3", "C3",
        "EEG C4-REF", "EEG C4-LE", "EEG C4", "C4",
        "EEG CZ-REF", "EEG CZ-LE", "EEG CZ", "CZ",
        "EEG T3-REF", "EEG T3-LE", "EEG T3", "T3",
        "EEG T4-REF", "EEG T4-LE", "EEG T4", "T4",
        "EEG T5-REF", "EEG T5-LE", "EEG T5", "T5",
        "EEG T6-REF", "EEG T6-LE", "EEG T6", "T6",
        "EEG O1-REF", "EEG O1-LE", "EEG O1", "O1",
        "EEG O2-REF", "EEG O2-LE", "EEG O2", "O2"]
        ch_necessary = ['FP1', 'FP2', 'F3', 'F4', 'FZ', 'F7', 'F8', 'P3', 'P4', 'PZ', 'C3', 'C4', 'CZ', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']
        #mappa dei nomi dai possibili usati ad uno standard semplice
        ch_mapper = {}
        for i in range(0, len(ch_map), 4):
            standard_name = ch_map[i + 3]
            for j in range(4):
                ch_mapper[ch_map[i + j]] = standard_name
        
        #rinomino i canali
        existing_channels = set(raw_obj.info['ch_names'])
        filtered_ch_mapper = {old: new for old, new in ch_mapper.items() if old in existing_channels}
        raw_obj.rename_channels(filtered_ch_mapper)
        ch_names = set(raw_obj.ch_names)

        #controllo che tutti i nomi dei canali siano presenti
        if set(ch_necessary).issubset(ch_names):
            raw_obj.pick(ch_necessary)
        else:
            raise RuntimeError("Channel Error")

def evaluate(data_dir, model, model_files, params, out_dir, cuts, graphs=False, f_extensions=['.edf']):
    """
    INPUT: dove trovare i dati e dove trovare i modelli
    Ricavo una lista di campioni lunghi tot
    poi su questi campioni calcolo l'errore di fase e la connettività per banda
    OUTPUT: Grafici salvati come png delle matrici di correlazione della connettività e della fase
    """
    Bands = ['delta','theta','alpha','low_beta','high_beta']

    path_list = utils.get_path_list(data_dir, f_extensions,False)
    model = load_models(model, model_files, params)
    origis = []
    recs = []
    for j,path in enumerate(path_list):
        raw = utils.get_raw(path, preload=True)
        check_channel_names(raw_obj=raw, verbose=False)
        raw = raw.get_data()
        raw = np.array(raw, np.float64)
        raw = raw[:,cuts[0]:cuts[1]]
        orig, rec, _ = get_orig_rec_latent(raw, model)
        origis.append(orig)
        recs.append(rec)
        if j == 2:
            break
    print('CHECKPOINT: Fine calcolo originali e ricostruzioni')
    
    
    cc = ConComparison()
    orig_pcc, orig_pvl, rec_pcc, rec_pvl = cc.work(origis, recs)

    pcc_mae = np.abs(orig_pcc - rec_pcc)
    pcc_mae = pcc_mae.mean(axis=(0,2,3))
    pvl_mae = np.abs(orig_pvl-rec_pvl)
    pvl_mae = pvl_mae.mean(axis=(0,2,3))

    print('CHECKPOINT: Fine comparazione connettività')

    pc = PhaseComparison()
    pc_mae = pc.work(origis, recs)

    print('CECKPOINT: Fine comparazione di fase')

    if graphs:
        assert orig_pcc.shape[0] == 1 and rec_pcc.shape[0] == 1, "solo un campione per fare i grafici"
        orig_pcc = orig_pcc[0]
        rec_pcc = rec_pcc[0]

        for j,data in enumerate([orig_pcc, orig_pvl]):
            j='orig' if j==0 else 'rec'
            fig, axis = plt.subplot(1, 5, figsize=(4*5,3))
            v_min = data.min
            v_max = data.max
            for i in range(5):
                ax = axis[i]
                im = ax.imshow(data[i], champ='coolwarm', origin='upper', vmin=v_min, vmax=v_max)
                ax.set_title(f'Banda {Bands[i]}')
                ax.set_xticks(np.arange(len(STANDARD_1020)))
                ax.set_xticklabels(STANDARD_1020, rotation=90)
                ax.set_yticks(np.arange(len(STANDARD_1020)))
                ax.set_yticklabels(STANDARD_1020)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
            fig.colorbar(im, cax=cbar_ax, )
            fig.supxlabel('Channels')
            fig.supylabel('Channels')
            path = os.path.join(out_dir,f'Rec_eval/heatmap{j}.png')
            plt.savefig(path)

    pears_is = []
    nrmse_s = []
    for o,r in zip(orig, rec):
        pears_is.append(pearson_index(o,r))
        nrmse_s.append(NRMSE(o,r))
        
    return pcc_mae, pvl_mae, pc_mae, np.mean(np.array(pears_is)), np.mean(np.array(nrmse_s))
