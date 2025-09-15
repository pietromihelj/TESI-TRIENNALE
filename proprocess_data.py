import mne
import os
import numpy as np
from utils import get_path_list, get_raw, stride_data
import gc
from deploy import load_models
from time import time
from tqdm import tqdm
import torch

#scale = 1.3e+8
scale = 2e+11
#scale = 0.0014065
BANDS =  [("delta", (1.0, 4.0)),
            ("theta", (4.0, 8.0)),
            ("alpha", (8.0, 13.0)),
            ("low_beta", (13, 20)),
            ("high_beta", (20, 30.0))]

model = load_models(model='VAEEG', save_files=['/u/pmihelj/models/models/VAEEG/delta_band.pth', '/u/pmihelj/models/models/VAEEG/theta_band.pth', '/u/pmihelj/models/models/VAEEG/alpha_band.pth', '/u/pmihelj/models/models/VAEEG/low_beta_band.pth', '/u/pmihelj/models/models/VAEEG/high_beta_band.pth'], params=[[8], [10], [12], [10], [10]])
#model = load_models(model='FastICA', save_files="/u/pmihelj/models/models/fast_ica/parallel/logcosh_FastICA_whole.pkl", params=[1])
#model = load_models(model='KernelPCA', save_files="/u/pmihelj/models/k_pca/rbf/10/13400_KernelPCA_whole.pkl", params=[1])
#model = load_models(model='KernelPCA', save_files="/u/pmihelj/models/k_pca/rbf/0.1/13400_KernelPCA_whole.pkl", params=[1])
#model = load_models(model='PCA', save_files="models/PCA_whole.pkl", params=[1])
print(f'Modello caricato: {model}')

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

#paths = get_path_list("/u/pmihelj/datasets/seizure_monochannel/seizure", f_extensions=['.edf'], sub_d=True)
paths = get_path_list("/u/pmihelj/datasets/seizure_monochannel/normal", f_extensions=['.edf'], sub_d=True)
print(len(paths))
#save_b = "/u/pmihelj/datasets/seiz_ds/FastICA/seizure"
#save_b = "/u/pmihelj/datasets/seiz_ds/FastICA/normal"
#save_b = "/u/pmihelj/datasets/seiz_ds/KernelPCA/normal"
#save_b = "/u/pmihelj/datasets/seiz_ds/KernelPCA/seizure"
save_v = "/u/pmihelj/datasets/seiz_ds/Vaeeg/normal"
#save_v = "/u/pmihelj/datasets/seiz_ds/Vaeeg/seizure"
print(f'Save_dir: {save_v}') 
os.makedirs(save_v, exist_ok=True)
#os.makedirs(save_v, exist_ok=True)
for j,path in enumerate(tqdm(paths)):
    start = time()
    raw = get_raw(path)
    eeg = raw.get_data()
    raw.close()
    preprocessed = preprocess(eeg, fs=256)
    out = []
    for _, (lf, hf) in BANDS:
        band = mne.filter.filter_data(preprocessed, 250, l_freq=lf, h_freq=hf, method='iir', verbose=False).astype(np.float32)
        clips = stride_data(band, 250,0)
        out.append(clips)
    out = np.transpose(np.array(out), (1,2,0,3))
    latent = []
    for ch in out:
        c_r, c_l = model.run(torch.tensor(ch, dtype=torch.float32))
        latent.append(c_l.detach().numpy())
    latent = np.array(latent)
    base_name = os.path.splitext(os.path.basename(path))[0]
    np.save(os.path.join(save_v, base_name + '.npy'), latent)
    print(f'Salvataggio in tempo: {time()-start}')
    del out, preprocessed
    gc.collect()
print('Finito')

"""
    preprocessed_b = stride_data(preprocessed, 250, 0)
    latent = []
    for ch in preprocessed_b:
        c_r, c_l = model.run(ch)
        latent.append(c_l)
    latent = np.array(latent)
    base_name = os.path.splitext(os.path.basename(path))[0]
    np.save(os.path.join(save_b, base_name + '.npy'), latent)
    del preprocessed_b, raw, eeg
    gc.collect()
print('dati salvati')
"""