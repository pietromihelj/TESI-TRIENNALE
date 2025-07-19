from sklearn.decomposition import PCA, FastICA
import numpy as np
import pickle
import os
import torch
from functools import wraps
from inspect import signature
from utils import check_type, type_assert
import warnings


z_dims = 50

def train_fast_ICA(in_dir, out_dir, start_time, end_time):
    #prendo il path dove salvero l'oggetto
    out_path = os.path.join(out_dir, "FastICA_whole.pkl")
    if not os.path.isdir(out_dir):
        raise Exception('Directory di output non esistente, creare la directory prima')
    #prendo il path ai dati di training
    tr_path = os.path.join(in_dir,"whole.npy")
    #prendo i dati di training
    tr_data = np.load(tr_path)[:,start_time:end_time]
    #calcolo fastica e lo salvo come oggetto
    with open(out_path, 'wb') as f:
        ica = FastICA(n_components=z_dims)
        ica.fit(tr_data)
        pickle.dump(ica,f)

def train_fast_ICA(in_dir, out_dir, start_time, end_time):
    #identico a sopra ma cambia l'algoritmo che uso in PCA
    out_path = os.path.join(out_dir, "PCA_whole.pkl")
    if not os.path.isdir(out_dir):
        raise Exception('Directory di output non esistente, creare la directory prima')
    tr_path = os.path.join(in_dir,"whole.npy")
    tr_data = np.load(tr_path)[:,start_time:end_time]
    with open(out_path, 'wb') as f:
        pca = PCA(n_components=z_dims)
        pca.fit(tr_data)
        pickle.dump(pca,f)


@type_assert(model=torch.nn.Module, ckpt_files=str)
def load_model(model, ckpt_file):
    #funzione per caricare un modello i cui pesi sono salvati in ckpt_file
    state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict["model"])
    return state_dict["auxiliary"]

@type_assert(model=torch.nn.Module)
def save_model(model, out_file, auxiliary=None):
    #salvare i pesi di un modello in un file esterno
    check_type("auxiliary", auxiliary, [dict, type(None)])
    data = dict()

    if auxiliary is None:
        data["auxiliary"] = {}
    else:
        data["auxiliary"] = auxiliary
    
    if isinstance(model, torch.nn.DataParallel):
        data["model"] = model.module.state_dict()
    else:
        data["model"] = model.state_dict()
    
    torch.save(data, out_file)

@type_assert(model=torch.nn.Module, n_gpus=int)
def init_model(model, n_gpus=0, ckpt_file=None):
    #inizializzatore del modello
    #check dei tipi
    check_type("ckpt_file", ckpt_file,[str, type(None)])
    #setto il device
    n_gpus = min(max(n_gpus,0), torch.cuda.device_count())
    device = "cuda" if n_gpus > 0 else "cpu"


    if isinstance(ckpt_file, str):
        if os.path.isfile(ckpt_file):
            #se il ckpt_file esiste allora carico il modello usandolo
            print("Initial model from %s" % ckpt_file)
            aux = load_model(model, ckpt_file)
        else:
            #se no inizializzo un modello con pesi casuali
            warnings.warn("The given checkpoint file is not found: %s. "
                          "Initial model randomly instead." % ckpt_file)
            aux = {}
    else:
        print("Initial model randomly")
        aux = {}

    #messagi di print del device 
    msg = "Assign model to %s device." % device

    if n_gpus > 0:
        msg += "Using %d gpus." % n_gpus
    
    print(msg)
    model.to(device)
    #se ho tante gpu devo salvare il modello in un oggetto DataParallel
    if n_gpus > 1:
        om = torch.nn.DataParallel(model, list(range(n_gpus)))
    else:
        om = model
    
    return om, aux, device

class train_VAEEG():
    def __init__(self, in_model, n_gpus, ckpt_file = None):
        #inizializzo il modello con la funzione preposta
        self.model, self.aux, self.device = init_model(in_model, n_gpus=n_gpus, ckpt_file=ckpt_file)

    @staticmethod
    def pearson_index(x, y, dim=-1):
        #calc olo manuale del coeff di correlazione di pearson lungo una specifica dimensione
        xy = x * y
        xx = x * x
        yy = y * y

        mx = x.mean(dim)
        my = y.mean(dim)
        mxy = xy.mean(dim)
        mxx = xx.mean(dim)
        myy = yy.mean(dim)

        r = (mxy - mx * my) / torch.sqrt((mxx - mx ** 2) * (myy - my ** 2))
        return r
    