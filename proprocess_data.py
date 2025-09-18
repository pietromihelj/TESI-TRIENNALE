import mne
import os
import numpy as np
from utils import get_path_list, get_raw, stride_data
import gc
from deploy import load_models
import mne
from tqdm import tqdm
import torch

BANDS =  [("delta", (1.0, 4.0)),
            ("theta", (4.0, 8.0)),
            ("alpha", (8.0, 13.0)),
            ("low_beta", (13, 20)),
            ("high_beta", (20, 30.0))]

model = load_models(model='VAEEG', save_files=['models/VAEEG/delta_band.pth', 'models/VAEEG/theta_band.pth', 'models/VAEEG/alpha_band.pth', 'models/VAEEG/low_beta_band.pth', 'models/VAEEG/high_beta_band.pth'], params=[[8], [10], [12], [10], [10]])
#model = load_models(model='FastICA', save_files="/u/pmihelj/models/models/fast_ica/parallel/logcosh_FastICA_whole.pkl", params=[1])
#model = load_models(model='KernelPCA', save_files="/u/pmihelj/models/k_pca/rbf/10/13400_KernelPCA_whole.pkl", params=[1])
#model = load_models(model='KernelPCA', save_files="/u/pmihelj/models/k_pca/rbf/0.1/13400_KernelPCA_whole.pkl", params=[1])
#model = load_models(model='PCA', save_files="models/logcosh_FastICA_whole.pkl", params=[1])
print(f'Modello caricato: {model}')

def preprocess(signal, amp_th = 400, m_len=1.0, drop = 60 ,fs=250):
    """
    INPUT: segnale EEG numpy [ch_num, temp_len]
    OUTPUT: numpy array segnale segmentato in clip da 1 secondo [ch_num, clip_num, clip_len]
    """
    #estraggo i dati e li scalo
    #sistemo la frequenza
    if fs != 250:
        ratio = 250/fs
        signal = mne.filter.resample(signal.astype(np.float64), ratio, verbose=False)
    signal = signal - signal.mean(axis=0, keepdims=True)

    mask = np.abs(signal) > amp_th
    window = 100  
    dilated_mask = np.zeros_like(signal, dtype=bool)
    for ch in range(signal.shape[0]):
        idx = np.where(mask[ch])[0]
        for i in idx:
            start = max(0, i - window)
            end = min(signal.shape[1], i + window + 1)
            dilated_mask[ch, start:end] = True
    clean_eeg = np.delete(signal, np.any(dilated_mask, axis=0), axis=1)
    return clean_eeg

#paths = get_path_list("/u/pmihelj/datasets/seizure_monochannel/seizure", f_extensions=['.edf'], sub_d=True)
paths = get_path_list("D:/sleep/Sleep_stage_N2", f_extensions=['.edf'], sub_d=True)
print(len(paths))
#save_b = "/u/pmihelj/datasets/seiz_ds/FastICA/seizure"
#save_b = "/u/pmihelj/datasets/seiz_ds/FastICA/normal"
#save_b = "/u/pmihelj/datasets/seiz_ds/KernelPCA/normal"
#save_b = "/u/pmihelj/datasets/seiz_ds/KernelPCA/seizure"
save_v = "D:/sleep/VAEEG/N2"
#save_v = "/u/pmihelj/datasets/seiz_ds/Vaeeg/seizure"
print(f'Save_dir: {save_v}') 
os.makedirs(save_v, exist_ok=True)
#os.makedirs(save_v, exist_ok=True)
total_len = 0
batch_size = 1   # secondi per batch (adatta alla tua GPU/RAM)
sfreq = 250
samples_per_batch = int(batch_size * sfreq)

for j, path in enumerate(tqdm(paths)):
    raw = get_raw(path)
    n_samples = raw.n_times

    base_name = os.path.splitext(os.path.basename(path))[0]
    latent_all = []

    for start in range(0, n_samples, samples_per_batch):
        stop = min(start + samples_per_batch, n_samples)
        # Carico solo il pezzo
        if stop <= start:  # batch vuoto
            print('start stop')
            continue

        eeg_chunk = raw.get_data(start=start, stop=stop)

        if eeg_chunk.shape[1] == 0:  # nessun campione
            print('chunk vuoto')
            continue
        # Preprocessa
        preprocessed = preprocess(eeg_chunk, fs=sfreq)

        if preprocessed.shape[1] == 0:  # ancora vuoto
            print('preprocess vuoto')
            continue

        out = []
        for _, (lf, hf) in BANDS:
            band = mne.filter.filter_data(
                preprocessed, sfreq, l_freq=lf, h_freq=hf,
                method='iir', verbose=False
            ).astype(np.float32)

            clips = stride_data(band, sfreq, 0)
            out.append(clips)

        out = np.transpose(np.array(out), (1, 2, 0, 3))

        latent = []
        for ch in out:
            c_r, c_l = model.run(torch.tensor(ch, dtype=torch.float32))
            latent.append(c_l.detach().cpu().numpy())
        latent = np.array(latent)

        if latent.shape[1] < batch_size:
            pad_width = batch_size - latent.shape[1]
            latent = np.pad(latent, ((0,0),(0,pad_width),(0,0)), mode="constant")
        elif latent.shape[1] > batch_size:
            latent = latent[:, :batch_size, :]

        latent_all.append(latent)
        del eeg_chunk, preprocessed, out, latent, band, clips
        gc.collect()

    raw.close()
    latent_all = np.concatenate(latent_all, axis=1)
    print(latent_all.shape)
    np.save(os.path.join(save_v, base_name + ".npy"), latent_all)
    del latent_all
    gc.collect()

print("Finito")

"""
    preprocessed_b = stride_data(preprocessed, 250, 0)
    latent = []
    for ch in preprocessed_b:
        c_r, c_l = model.run(ch)
        latent.append(c_l)
    latent = np.array(latent)    
    base_name = os.path.splitext(os.path.basename(path))[0]
    np.save(os.path.join(save_b, base_name + '.npy'), latent)
    del preprocessed_b, raw, eeg, latent
    gc.collect()
print(f'Dati salvati, Lunghezza: {total_len/250/60/60}')

"""


