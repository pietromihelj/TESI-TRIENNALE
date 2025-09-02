import numpy as np
import pywt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
import utils
from deploy import get_orig_rec_latent, load_models,
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error

STANDARD_1020 =  ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'CZ', 'C3', 'C4', 'PZ', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']

def NRMSE(x,y):
    #calcolo manuale dell'nmsre
    mean_x = np.mean(x)
    if mean_x == 0:
        return np.inf
    rmse = root_mean_squared_error( x, y)
    nrmse = rmse / mean_x
    return np.abs(nrmse)

class PhaseComparison():
    def __init__(self, orig_num):
        self.BANDS = {"delta": (1.0, 4.0),
                      "theta": (4.0, 8.0),
                      "alpha": (8.0, 13.0),
                      "low_beta": (13, 20),
                      "high_beta": (20, 30.0)}
        self.orig_num = orig_num
    
    def compare_mo_wa_ps(self, orig, rec, count, length, l_freq=1, h_freq=30, fs=250):
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
        coeff_rec, freq = pywt.cwt(rec, wavelet='cmor1.5-1.0', scales=scales, sampling_period=1/fs)
        rec_phase = np.angle(coeff_rec)
        coeff_orig, freq = pywt.cwt(orig, wavelet='cmor1.5-1.0', scales=scales, sampling_period=1/fs)
        orig_phase = np.angle(coeff_orig)

        #calcolo la media della differenza di fase per banda
        return np.mean(np.abs(np.angle(np.exp(1j*(orig_phase - rec_phase)))))
    
    def work(self, orig, rec):
        """
        INPUT: lista di segnali EEG
        OUTPUT: Dataframe contenente l'errore medio di fase per banda
        """
        #orig,rec hanno forma [ch_num, band_num, temp_len]
        res = []
        #per ogni coppia di valori calcolo la media per canale
        print('CHECKPOINT: comparazione di fase')
        counter = 0
        for o,r in tqdm(zip(orig, rec)):
            ch_res = []
            for ch_o, ch_r in zip(o,r):
                ch_res.append(self.compare_mo_wa_ps(ch_o.reshape(-1), ch_r.reshape(-1), count=counter, length=len(orig)))
            ch_res = np.mean(ch_res, axis=0)
            res.append(ch_res)
        #calcolo poi la media tra le coppie per banda
        return np.mean(res) 


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
    

def evaluate(data_dir, model, model_files, params, cuts, f_extensions=['.edf']):
    """
    INPUT: dove trovare i dati e dove trovare i modelli
    Ricavo una lista di campioni lunghi tot
    poi su questi campioni calcolo l'errore di fase e la connettività per banda
    OUTPUT: Grafici salvati come png delle matrici di correlazione della connettività e della fase
    """

    path_list = utils.get_path_list(data_dir, f_extensions,False)
    model = load_models(model, model_files, params)
    origis = []
    recs = []
    for path in path_list:
        raw = utils.get_raw(path, preload=True)
        utils.check_channel_names(raw_obj=raw, verbose=False)
        raw = raw.get_data()
        raw = np.array(raw, np.float64)
        raw = raw[:,cuts[0]:cuts[1]]
        orig, rec, _ = get_orig_rec_latent(raw, model)
        origis.append(orig)
        recs.append(rec)
    print('CHECKPOINT: Fine calcolo originali e ricostruzioni')
    
    cc = ConComparison()
    orig_pcc, orig_pvl, rec_pcc, rec_pvl = cc.work(origis, recs)

    pcc_mae = np.abs(orig_pcc - rec_pcc)
    pcc_mae = pcc_mae.mean(axis=(0,2,3))
    pvl_mae = np.abs(orig_pvl-rec_pvl)
    pvl_mae = pvl_mae.mean(axis=(0,2,3))

    print('CHECKPOINT: Fine comparazione connettività')

    pc = PhaseComparison(orig_num=len(origis))
    pc_mae = pc.work(origis, recs)

    print('CECKPOINT: Fine comparazione di fase')

    pears_is = []
    nrmse_s = []
    for o,r in zip(orig, rec):
        print(o.shape,r.shape)
        pears_is.append(pearsonr(o.flatten(),r.flatten()))
        nrmse_s.append(NRMSE(o.flatten(),r.flatten()))
    
    print('CHECKPOINT: Inizio calcolo ampiezza media')
    ch_ampl = [[],[]]
    for o,r in zip(orig, rec):
        ch_ampl[0].append(np.mean(np.abs(o),axis=1))
        ch_ampl[1].append(np.mean(np.abs(r), axis=1))
    
    return pcc_mae[0], pvl_mae[0], pc_mae, np.mean(np.array(pears_is)), np.mean(np.array(nrmse_s)), np.mean(np.stack(ch_ampl[0]), axis=0), np.mean(np.stack(ch_ampl[1], axis=0))
