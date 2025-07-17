from utils import check_type, get_raw, find_artefacts_2d, merge_continuous_artifacts
import gc
import numpy as np
import mne
import pandas as pd
import random
import os
import portion as P
class data_gen():
    _SFREQ = 250.0

    max_clips = 500

    _BANDS = {"whole": (1.0, 30.0),
              "delta": (1.0, 4.0),
              "theta": (4.0, 8.0),
              "alpha": (8.0, 13.0),
              "low_beta": (13, 20),
              "high_beta": (20, 30.0)}
    
    def __init__(self, f_name, d_out, out_prefix):
        self.f_name = f_name
        self.d_out = d_out
        self.out_prefix = out_prefix
        self.data, self.ch_names = self.get_filtered_data(verbose = False)
    
    @staticmethod
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
        
    def get_filtered_data(self, verbose):

        #prendo l'oggetto raw contenente l'eeg
        raw = get_raw(self.f_name, verbose)
        #controllo la presenza dei canali
        self.check_channel_names(raw,verbose)
        #carico direttamente i dati grezzi per modifica in place
        raw.load_data(verbose)
        #modifico la frequenza di campionamento
        raw.resample(self._SFREQ, verbose=verbose)
        #riferisco alla media per istante
        raw.set_eeg_reference(ref_channels='average', verbose=verbose)
        #prendo una copia dei dati grezzi
        data = raw.get_data()
        filter_results = {}
        
        #filtro il segnale in 5 bande
        for key,(lf,hf) in self._BANDS.items():
            
            filter_results[key] = mne.filter.filter_data(data, self._SFREQ, l_freq=lf, h_freq=hf, filter_length='auto', fir_design='firwin',verbose=verbose).astype(np.float32)

        ch_names = raw.ch_names
        
        #pulisco la memoria
        del raw, data
        gc.collect()
        return filter_results, ch_names
    
    @staticmethod
    def filter_intervalset(input_intervalset, threshold):
        return [i for i in input_intervalset if i.upper - i.lower > threshold]

    def save_final_data(self, seg_len = 5.0, amp_th = 400, merge_len = 1.0, drop = 60.0):
        #carico il segnale completo
        data = self.data['whole']
        
        #setto le treashold necessarie
        m_th = int(merge_len*self._SFREQ)
        s_th = int(seg_len*self._SFREQ)
        start = int(drop*self._SFREQ)
        end = data.shape[1] - int(drop * self._SFREQ)
        #creo un intervallo
        whole_r = P.closed(start, end)
        #creo la lista lista delle posizioni degli artefatti
        flag = np.abs(data) * 1e6 > amp_th
        #estraggo gli intervallli contenenti gli artefatti
        art = find_artefacts_2d(flag)#art è una lista di liste di intervalli
        #unisco gli artefatti vicini
        merged_art = [merge_continuous_artifacts(s, m_th) for s in art]#s è una lista di intervalli per uno specifico canale
        #merged_art è una lista di liste di intervalli
        #elimino i segmenti con artefatti da whole_r
        c_clean = [whole_r - s for s in merged_art]
        print(len(c_clean))
        print(type(c_clean))
        #tengo solo i campioni più lunghi di threashold
        keep_clean = [self.filter_intervalset(s, s_th) for s in c_clean]

        out = []

        #per ogni canale e set di intervalli associati
        for idx, item_set in enumerate(keep_clean, 0):
            #per ogni intervallo
            for item in item_set:
                #suddivido in segmenti di dimensione s_th
                tmp = np.arange(item.lower, item.upper + 1, s_th)
                if len(tmp) > 1:
                    for idj in range(len(tmp) - 1):
                        out.append([idx, self.ch_names[idx], tmp[idj], tmp[idj + 1]])
        #creo un dataframe con codice canale, nome, e punto di inizio e fine
        df_clean = pd.DataFrame(out, columns=["idx", "ch_names", "clip_start", "clip_stop"])

        #unifico in dati in un array 3d
        udata = np.stack([self.data["whole"],
                          self.data["delta"],
                          self.data["theta"],
                          self.data["alpha"],
                          self.data["low_beta"],
                          self.data["high_beta"]], axis=0) * 1.0e6
        
        #lista dei segmenti
        seg_list = list(zip(df_clean.idx, df_clean.clip_start, df_clean.clip_stop))

        if len(seg_list) >= 200:
            if len(seg_list) > self.max_clips:
                seg_list = random.sample(seg_list, self.max_clips)
            
            outputs = []
            for p in seg_list:
                idx, start, stop = p
                tmp = udata[:, idx, start:stop]
                outputs.append(tmp)
            
            outputs = np.stack(outputs, axis=0)
            out_file = os.path.join(self.d_out, "%s.npy" % self.out_prefix)
            np.save(out_file, outputs)