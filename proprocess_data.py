import mne
import os
import numpy as np
from utils import get_path_list, get_raw, stride_data
import gc
from deploy import load_models
from time import time
from tqdm import tqdm

scale = 1.3e+8
BANDS =  [("delta", (1.0, 4.0)),
            ("theta", (4.0, 8.0)),
            ("alpha", (8.0, 13.0)),
            ("low_beta", (13, 20)),
            ("high_beta", (20, 30.0))]

#model_vaeeg = load_models(model='VAEEG', save_files=['models/VAEEG/delta_band.ckpt', 'models/VAEEG/theta_band.ckpt', 'models/VAEEG/alpha_band.ckpt', 'models/VAEEG/low_beta_band.ckpt', 'models/VAEEG/high_beta_band.ckpt'], params=[[8], [10], [12], [10], [10]])
model_k_pca = load_models(model='KernelPCA', save_files='models/k_pca/rbf/10/13400_KernelPCA_whole.pkl', params=[1])
print('Modelli caricati')

def preprocess(signal, amp_th = 200, m_len=1.0, drop = 60 ,fs=250):
    """
    INPUT: segnale EEG numpy [ch_num, temp_len]
    OUTPUT: numpy array segnale segmentato in clip da 1 secondo [ch_num, clip_num, clip_len]
    """
    #estraggo i dati e li scalo
    signal = signal * scale
    #sistemo la frequenza
    if fs != 250:
        ratio = 250/fs
        signal = mne.filter.resample(signal.astype(np.float64), ratio, verbose=False)
    signal = signal - signal.mean(axis=0, keepdims=True)

    mask = np.abs(eeg) > amp_th
    window = 100  
    dilated_mask = np.zeros_like(eeg, dtype=bool)
    for ch in range(eeg.shape[0]):
        idx = np.where(mask[ch])[0]
        for i in idx:
            start = max(0, i - window)
            end = min(eeg.shape[1], i + window + 1)
            dilated_mask[ch, start:end] = True
    clean_eeg = np.delete(eeg, np.any(dilated_mask, axis=0), axis=1)
    return clean_eeg

paths = get_path_list("D:/nmt_scalp_age_dataset", f_extensions=['.edf'], sub_d=True)
print(len(paths))
save_b = "C:/Users/Pietro/Desktop/age_eeg/rbf_10"
#save_v = "C:/Users/Pietro/Desktop/seizure_dataset/vaeeg_latent/seizure"
os.makedirs(save_b, exist_ok=True)
#os.makedirs(save_v, exist_ok=True)
for path in tqdm(paths):
    start = time()
    raw = get_raw(path)
    eeg = raw.get_data()
    raw.close()
    preprocessed = preprocess(eeg, fs=256)
    preprocessed_b = stride_data(preprocessed, 250, 0)
    latent = []
    for ch in preprocessed_b:
        c_r, c_l = model_k_pca.run(ch)
        latent.append(c_l)
    latent = np.array(latent)
    base_name = os.path.splitext(os.path.basename(path))[0]
    np.save(os.path.join(save_b, base_name + '.npy'), preprocessed_b)
    del preprocessed_b, raw, eeg
    gc.collect()

"""
    out = []
    for _, (lf, hf) in BANDS:
        band = mne.filter.filter_data(preprocessed, 250, l_freq=lf, h_freq=hf, method='iir', verbose=False).astype(np.float32)
        clips = stride_data(band, 250,0)
        out.append(clips)
    out = np.transpose(np.array(out), (1,2,0,3))
    latent = []
    for ch in out:
        c_r, c_l = model_vaeeg.run(ch)
        latent.append(c_l)
    latent = np.array(latent)
    base_name = os.path.splitext(os.path.basename(path))[0]
    np.save(os.path.join(save_v, base_name + '.npy'), latent)
    print(f'Salvataggio in tempo: {time()-start}')
    del out, preprocessed
    gc.collect()
"""