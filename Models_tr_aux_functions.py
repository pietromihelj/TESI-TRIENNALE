from sklearn.decomposition import PCA, FastICA
import numpy as np
import pickle
import os
import torch
from utils import check_type, type_assert
import warnings
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import itertools as it
import time
import model as md
import matplotlib.pyplot as plt


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

def get_signal_plot(input_y, output_y, sfreq=250, fig_size=(8,5)):
    if not (isinstance(input_y, np.ndarray) and input_y.ndim == 1 and input_y.shape == output_y.shape):
        raise RuntimeError("y is not supported.")
    
    fig = plt.figure(figsize=fig_size)
    ax = plt.subplot(111)
    xt = np.range(0,input_y.shape[0])/sfreq
    ax.plot(xt, input_y, label='input')
    ax.plot(xt, output_y, label='output')
    ax.legend(fontsize='large')
    ax.grid(axis='x', linstile='-.', linewidth=1, wich='both')
    ax.set_ylabel("amp (uV)", fontdict={"fontsize": 15})
    ax.set_xlabel("time (s)", fontdict={"fontsize": 15})
    ax.tick_params(labelsize=15)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return img


def get_signal_plots(input_y, output_y, sfreq, fig_size=(8,5)):
    if not (isinstance(input_y, np.ndarray) and input_y.ndim == 2 and input_y.shape == output_y.shape):
        raise RuntimeError('y is not supported')
    #per ogni coppia in-out creo il grafico con get_signal_plot, poi li rendo una lista e li concateno
    out = map(lambda a: get_signal_plot(a[0], a[1], sfreq, fig_size), zip(input_y, output_y))
    return np.stack(list(out), axis=0)

def batch_imgs(input_y, output_y, sfreq, num, n_row, fig_size=(8, 5)):
    z = get_signal_plots(input_y[0:num, :], output_y[0:num, :], sfreq, fig_size)
    img = make_grid(torch.tensor(np.transpose(z, (0, 3, 1, 2))), nrow=n_row, pad_value=0, padding=4)
    a = img.numpy()
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)))

def save_loss_per_line(file, line, header):
    if os.path.isfile(file):
        #leggo il file e controllo che sia vuoto
        with open(file, "r") as fi:
            dat = [line.strip() for line in fi.readlines() if line.strip() != ""]

        # se il file è vuoto o ha un header sbagliato lo scrivo/sovrascrivo
        if len(dat) == 0 or dat[0] != header:
            with open(file, "w") as fo:
                print(header, file=fo)
                print(line, file=fo)
        else:
            #se no scrivo solo la riga
            with open(file, "a") as fo:
                print(line, file=fo)
    else:
        #se il file non esiste lo creo e lo popolo
        with open(file, "w") as fo:
            print(header, file=fo)
            print(line, file=fo)

 
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
    
    def train(self, input_loader, model_dir, n_epochs, lr, beta, n_print):
        #creo la directori dove salvare le informazioni sul modello
        summary_dir = os.path.join(model_dir,'save')
        if not os.path.isdir(summary_dir):
            os.makedirs(summary_dir)
        #creo il csv dove salvero le loss
        loss_file = os.path.join(model_dir, 'train_loss.csv')
        writer = SummaryWriter(summary_dir)

        #carico il punto a cui sono arrivato con il training
        current_epoch = self.aux.get("current_epoch", 0)
        current_step = self.aux.get("current_step",0)

        #creo l'optimizer
        optimizer = torch.optim.RMSprop(it.chain(self.model.encoder.parameters(), self.model.decoder.parameters()), lr=lr)

        #setto il modello in modalità train
        self.model.train()
        start_time = time.time()

        #inizio il training
        for epoch in range(n_epochs):
            current_epoch = current_epoch+1
            for idx, input in enumerate(input_loader, 0):
                current_step = current_step+1
                #passo i dati nel modello
                mu, log_var, x_rec = self.model(input)
                kld = md.kl_loss(mu,log_var)
                rec = md.recon_loss(input,x_rec)
                loss = beta*kld+rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if current_epoch % n_print == 0:
                writer.add_scalar('loss', loss, current_epoch)
                writer.add_scalar('kld_loss', kld, current_epoch)
                writer.add_scalar('rec_loss', rec, current_epoch)

                error = input-x_rec
                error = error.abs().mean()
                writer.add_scalar('MAE_error', error, current_epoch)

                pr = self.pearson_index(input,x_rec)
                writer.add_scalar('pearsonr', pr.mean(), current_epoch)

                cycle_time = (time.time()-start_time)/n_print

                values = (current_epoch, current_step, cycle_time,
                              loss.to(torch.device("cpu")).detach().numpy(),
                              kld.to(torch.device("cpu")).detach().numpy(),
                              rec.to(torch.device("cpu")).detach().numpy(),
                              error.to(torch.device("cpu")).detach().numpy(),
                              pr.mean().to(torch.device("cpu")).detach().numpy())
                names = ["current_epoch", "current_step", "cycle_time", "loss", "kld_loss", "rec_loss","error", "pr"]
                print("[Epoch %d, Step %d]: (%.3f s / cycle])\n""  loss: %.3f; kld_loss: %.3f; rec: %.3f;\n""  mae error: %.3f; pr: %.3f.\n"% values)
                
                img = batch_imgs(input.to(torch.device("cpu")).detach().numpy()[:, 0, :], x_rec.to(torch.device("cpu")).detach().numpy()[:, 0, :],250, 4, 2, fig_size=(8, 5))
                writer.add_images('signal', img, current_epoch, dataformats='HWC')
                
                start_time = time.time()
                
                n_float = len(values)-2
                fmt_str = "%d,%d" + ",%.3f" * n_float
                save_loss_per_line(loss_file, fmt_str % values, ",".join(names))
            
            out_ckpt_file = os.path.join(model_dir, "ckpt_epoch_%d.ckpt" % current_epoch)
            save_model(self.model, out_file=out_ckpt_file,auxiliary=dict(current_step=current_step,current_epoch=current_epoch))
        writer.close()