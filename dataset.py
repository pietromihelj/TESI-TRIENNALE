import json
import os
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count
from utils import get_path_list
from tqdm import tqdm
import torch.utils.data

def merge_data(path_list, out_dir, seed=0, clip_len=250):
    #setto il seed per la riproducibilità
    np.random.seed(seed)
    #nomino le bande
    band_names = ["whole", "delta", "theta", "alpha",  "low_beta", "high_beta"]
    #prendo il numero di processi
    n_jobs = cpu_count()
    #leggo i file
    data = Parallel(n_jobs=n_jobs, backend='loky')(delayed(np.load)(f) for f in tqdm(path_list))
    #per motivi di poco spazio sulla mia macchina devo fare in questo punto il sempling di segmenti di 1 secondo da quelli da 5
    data_cut = []
    for cmp in data:
        cmp_cut = []
        for clip in cmp:
            n_len = clip.shape[1]
            if n_len < clip_len:
                print("clip di dimensione: ",n_len)
                continue  # scarta clip troppo corte
            start = np.random.randint(0, n_len - clip_len)
            segment = clip[:, start:(start+clip_len)]
            if segment.shape[1] == clip_len:
                cmp_cut.append(segment)
        if len(cmp_cut) > 0:
            cmp_cut = np.stack(cmp_cut, axis=0)  
            data_cut.append(cmp_cut)
    print(data_cut[0].shape)
    
    #concateno lungo la dimensione del numero dei campioni ottenendo [n_campioni totale, 6, lunghezza campione]
    data_cut = np.concatenate(data_cut, axis=0)
    np.random.shuffle(data_cut)

    #salvo 6 file (1 per banda) con la forma [n_campioni totali, lunghezza campione]
    for i, name in enumerate(band_names):
        print("save %s to %s" % (name, out_dir))
        #creo il path del file
        out_files = os.path.join(out_dir, name + '.npy')
        #creo il numpy array del file da salvare
        sx = data_cut[:,i,:]
        np.save(out_files, sx)

def make_save_dataset(f_dir, out_dir, ratio=0.2, seed=0,clip_len=250):
    #se non esiste creo la directory di output
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    #prendo i path ai file da leggere
    path_list = get_path_list(d_path=f_dir, f_extensions=['.npy']).tolist()
    print("paths presi")
    #creo il file contenente i paths
    paths_file = os.path.join(out_dir, "dataset_paths.json")
    
    random.shuffle(path_list)

    #suddivido i paths in training e test dopo averli mischiati
    n_tr = int((1.0-ratio)*len(path_list))
    tr_paths = path_list[0:n_tr]
    te_paths = path_list[n_tr:]
    print("create liste di train e test dei path")

    #popolo il file dei paths all'interno della output directory
    with open(paths_file,"w") as fo:
        json.dump({"train":tr_paths, "test":te_paths}, fp=fo, indent=1)

    #creo la directory di test se non esiste
    te_dir = os.path.join(out_dir,"test")
    if not os.path.isdir(te_dir):
        os.makedirs(te_dir)
    
    #creo la directory di train se non esiste
    tr_dir = os.path.join(out_dir,"train")
    if not os.path.isdir(tr_dir):
        os.makedirs(tr_dir)
    print("create directory di test e train")
    #popolo la directory di test
    print("inizio salvataggio file di test")
    merge_data(te_paths, te_dir, seed, clip_len)
    #popolo la directory di train
    print("inizio salvataggio file di train")
    merge_data(tr_paths,tr_dir, seed, clip_len)
    print("DONE!")

class ClipDS(torch.utils.data.Dataset):
    def __init__(self, data_dir, band_name, clip_len=250):
        #creo il path al file
        file_paths = os.path.join(data_dir,"%s.npy"%band_name)
        print('path: ',file_paths) 
        #carico il file contenente tutti i campioni di tutte le bande di tutti i soggetti messi in fila
        self.data = np.load(file_paths)

        self.band = band_name
        #prendo il numero di campioni e la sua dimensione
        self.n_item, self.n_len = self.data.shape
        #setto la lunghezza del dato voluta
        self.clip_len = clip_len
        print("elementi:",self.n_item)
        print("lunghezza campione:",self.n_len)
        print("lunghezza voluta clip:", self.clip_len)

    def __getitem__(self, index):
        #prendo un punto di inizio del dat in maniera tale che rimanga possibile prendere un 
        #segmento lungo clip_len
        if self.n_len == self.clip_len:
           idx=0
        else:
           idx = np.random.randint(0,self.n_len-self.clip_len)
        #estraggo un segmento lungo clip_len
        x = self.data[index:index+1, idx: idx+self.clip_len]
        return x
    
    def __len__(self):
        return self.n_item
