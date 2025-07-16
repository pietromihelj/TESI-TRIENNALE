from utils import check_type, get_raw
import gc
import numpy as np
import mne

class data_gen():
    _SFREQ = 250.0

    max_clips = 500
    _l_trans_bandwidth = 0.1
    _h_trans_bandwidth = 0.1

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

        ch_mapper = {}
        for i in range(0, len(ch_map), 4):
            standard_name = ch_map[i + 3]
            for j in range(4):
                ch_mapper[ch_map[i + j]] = standard_name
        
        raw_obj.rename_channels(ch_mapper)
        ch_names = set(raw_obj.ch_names)
        ch_neccessary = set(ch_mapper.values())
        if set(ch_neccessary).issubset(ch_names):
            raw_obj.pick_channels(ch_neccessary, ordered=True)
        else:
            raise RuntimeError("Channel Error")
        
    def get_filtered_data(self, verbose):

        raw = get_raw(self.f_name, verbose)
        self.check_channel_names(raw,verbose)
        raw.load_data(verbose)
        raw.resample(self._SFREQ, verbose)
        raw.set_eeg_reference(ref_channels='average', verbose=verbose)
        data = raw.get_data()
        filter_results = {}
        
        for key,(lf,hf) in self._BANDS.items():
            filter_results[key] = mne.filter.filter_data(data,self._SFREQ,l_freq=lf,h_freq=hf,l_trans_freq=self._l_trans_bandwidth, h_trans_freq=self._h_trans_bandwidth,verbose=verbose).astype(np.float32)

        ch_names = raw.ch_names
        
        del raw, data
        gc.collect()
        return filter_results, ch_names
    
    def save_final_data(self, seg_len = 5.0, amp_th = 400, merge_len = 1.0, drop = 60.0):
        data = self.data[whole]
        
