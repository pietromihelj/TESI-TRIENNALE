import numpy as np 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, Sampler
from torch.nn import CrossEntropyLoss
from time import time
from utils import get_path_list, stride_data
import os
import random
from sklearn import metrics
import gc
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class LSTM_model(nn.Module):
    def __init__(self, dim=200):
        super(LSTM_model, self).__init__()
        # self.laynorm = nn.LayerNorm((30, 4, 50))
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.lstm_0 = nn.LSTM(dim, 512, num_layers=2, batch_first=True,  bidirectional=False, dropout=0.2)
        self.layernorm_0 = nn.LayerNorm((30, 512))
        self.lstm_1 = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=False, dropout=0.2)
        self.layernorm_1 = nn.LayerNorm((30, 256))
        self.lstm_2 = nn.LSTM(256, 128, num_layers=2, batch_first=True,  bidirectional=False, dropout=0.3)
        self.layernorm_2 = nn.LayerNorm((30, 128))
        self.lstm_3 = nn.LSTM(128, 32, num_layers=2, batch_first=True,  bidirectional=False, dropout=0.1)
        self.layernorm_3 = nn.LayerNorm((30, 32))
        self.linear = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1), nn.Linear(32 * 30, 512), nn.ReLU(), nn.Dropout(p=0.3),
                                    nn.Linear(512, 64), nn.ReLU(), nn.Dropout(p=0.1),
                                    nn.Linear(64, 5))
    
    def forward(self, data):
        # data = self.laynorm(data)
        data = self.flatten(data)
        data, _ = self.lstm_0(data)
        data = self.layernorm_0(data)
        data, _ = self.lstm_1(data)
        data = self.layernorm_1(data)
        data, _ = self.lstm_2(data)
        data = self.layernorm_2(data)
        data, _ = self.lstm_3(data)
        data = self.layernorm_3(data)
        res = self.linear(data)
        return res

class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, bias=True):
        super(Conv1dLayer, self).__init__()

        total_p = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        left_p = total_p // 2
        right_p = total_p - left_p

        self.conv = nn.Sequential(nn.ConstantPad1d((left_p, right_p), 0),
                                  nn.Conv1d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride, dilation=dilation,
                                            bias=bias))

    def forward(self, x):
        return self.conv(x)
    
class HeadLayer(nn.Module):
    """
    Multiple paths to process input data. Four paths with kernel size 5, 7, 9, respectively.
    Each path has one convolution layer.
    """

    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(HeadLayer, self).__init__()

        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4, but got: %d" % out_channels)

        unit = out_channels // 4

        self.conv1 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=9, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv2 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=7, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv3 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=5, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv4 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit,
                                               kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))
        

        self.conv5 = nn.Sequential(Conv1dLayer(in_channels=out_channels, out_channels=out_channels,
                                               kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm1d(out_channels),
                                   nn.LeakyReLU(negative_slope))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv5(out)
        return out

class SleepStage_Model(nn.Module):
    def __init__(self, target_channels=4):
        super(SleepStage_Model, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.headlayer = HeadLayer(7, target_channels)
        self.lstm = LSTM_model(200)

    def forward(self, x):
        x = self.flatten(x)
        x = self.headlayer(x)
        x = torch.split(x, 50, dim=-1)
        x = torch.stack(x, dim=1)
        x = self.lstm(x)

        return x
   
def get_metrics(predict, label, class_num):
    metric_cm = metrics.confusion_matrix(label, predict, labels=list(range(class_num)))
    
    # Normalizzazione riga per riga, evitando divisioni per zero
    row_sums = metric_cm.sum(axis=1, keepdims=True)
    metric = np.divide(metric_cm, row_sums, out=np.zeros_like(metric_cm, dtype=float), where=row_sums!=0)
    
    acc_per_class = np.diagonal(metric)
    total_acc = np.trace(metric_cm) / np.sum(metric_cm) if np.sum(metric_cm) != 0 else 0.0
    
    # Restituisce matrice e array di accuracy per classe + totale
    return metric_cm, np.round(np.append(acc_per_class, total_acc), 3)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.ModuleList):
            for m_ in m:
                initialize_weights(m_)
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for name, parma in m.named_parameters():
                if name.find('weight') != -1 :
                    nn.init.xavier_normal_(parma)
                elif name.find('bias_ih') != -1:
                    nn.init.constant_(parma, 1)
    return model

class SleepDataset(Dataset):
    def __init__(self, N1_file, N2_file, N3_file, R_file, W_file):
        self.files = []
        self.labels = []
        for i, paths in enumerate([N1_file, N2_file, N3_file, R_file, W_file]):
            for path in paths:
                self.files.append(path)
                self.labels.append(i)
                if i == 4:
                    print(np.load(path).shape)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        clips = np.load(path, allow_pickle=True)
        clips = stride_data(np.transpose(clips,(0,2,1)), 30, 0)
        clips = np.transpose(clips, (2,3,0,1))
        # Qui ritorni tutti i clip in questo file, eventualmente puoi restituire uno solo per campione
        return torch.tensor(clips[0], dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class AtLeastOnePerClassSampler(Sampler):
    def __init__(self, class_indices, batch_size):
        self.class_indices = {c: idxs.copy() for c, idxs in class_indices.items()}
        self.batch_size = batch_size
        self.num_classes = len(class_indices)

        # Numero massimo di batch possibile (limitato dalla classe più piccola)
        total_samples = sum(len(v) for v in class_indices.values())
        self.num_batches = total_samples // batch_size

    def __iter__(self):
        # Copie locali da modificare
        pools = {c: idxs.copy() for c, idxs in self.class_indices.items()}
        for c in pools:
            random.shuffle(pools[c])

        for _ in range(self.num_batches):
            batch = []

            # 1 campione per ogni classe
            for c, idxs in pools.items():
                if len(idxs) == 0:  # se finiti, riciclo random
                    idx = random.choice(self.class_indices[c])
                else:
                    idx = idxs.pop()
                batch.append(idx)

            # Riempire batch con altri campioni random
            remaining = self.batch_size - self.num_classes
            all_indices = sum(self.class_indices.values(), [])
            batch += random.choices(all_indices, k=remaining)

            random.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self):
        return self.num_batches * self.batch_size

def pad_collate(batch):
    data, labels = zip(*batch)
    # Trova la dimensione massima dei canali o timesteps
    max_ch = max([d.shape[1] for d in data])
    max_len = max([d.shape[2] for d in data])
    
    padded_data = []
    for d in data:
        pad_ch = max_ch - d.shape[1]
        pad_len = max_len - d.shape[2]
        padded = torch.nn.functional.pad(d, (0, pad_len, 0, pad_ch))
        padded_data.append(padded)
    
    return torch.stack(padded_data), torch.tensor(labels)

def train_model(model, n_epoch, trainloader, testloader, lr, save_dir, device='cuda', steps = 1):

    model = SleepStage_Model().to(device)
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size=100, gamma=0.1)
    loss_fn = CrossEntropyLoss(weight=torch.tensor([1,1,1,1,1], device=device).float(),reduction="mean")
    total_losses = []
    for e in range(n_epoch):
        start = time()
        model.train()
        tr_losses = []
        tr_accs = []

        for tr_data, labels in trainloader:
            tr_data = tr_data.transpose(1,2)
            tr_data, labels = tr_data.float().to(device), labels.long().to(device)
            print(tr_data.shape)
            optimizer.zero_grad()
            pred = model(tr_data)
            loss = loss_fn(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            tr_losses.append(loss.item())
            prediction = torch.argmax(pred, dim=1)
            _, accs = get_metrics(prediction.cpu().numpy(), labels.cpu().numpy(), class_num=5)
            tr_accs.append(accs[-1])
            torch.cuda.empty_cache()

        scheduler.step()
        total_losses.append(np.mean(tr_losses))
        if e % steps == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {e}, Loss: {np.mean(total_losses)}, Accuracy: {np.mean(tr_accs)}, Learning Rate: {lr}, Time: {time()-start}')
            torch.save({'epoch': e,'model_state_dict': model.state_dict(),'loss': np.mean(total_losses),'accuracy': np.mean(tr_accs)}, save_dir + f'/saves/ckpt_{e}.pt')
    
    model.eval()
    te_CMs = []
    stage_accs = [] 
    for te_data, labels in testloader:
        te_pred = model(te_data.float().to(device))
        te_pred = torch.argmax(te_pred, dim=1).cpu().detach().numpy()
        te_cm, te_accs = get_metrics(te_pred, labels.to('cpu').detach().numpy(), class_num=5)
        te_CMs.append(te_cm)
        stage_accs.append(te_accs)
    stage_accs = np.mean(np.array(stage_accs), axis=0)

    print(f"Test accuracy per stage: N1 {stage_accs[0]}, N2 {stage_accs[1]}, N3 {stage_accs[2]}, R {stage_accs[3]}, W {stage_accs[4]}, Total {stage_accs[5]}")

    return np.mean(te_CMs, axis=0), stage_accs, total_losses


#N1_dir = "/u/pmihelj/datasets/sleep/VAEEG/N1/"
#N2_dir = "/u/pmihelj/datasets/sleep/VAEEG/N2/"
#N3_dir = "/u/pmihelj/datasets/sleep/VAEEG/N3/"
#R_dir = "/u/pmihelj/datasets/sleep/VAEEG/R/"
#W_dir = "/u/pmihelj/datasets/sleep/VAEEG/W/"

#N1_paths = get_path_list(N1_dir, f_extensions=['.npy'], sub_d=True)
#N2_paths = get_path_list(N2_dir, f_extensions=['.npy'], sub_d=True)
#N3_paths = get_path_list(N3_dir, f_extensions=['.npy'], sub_d=True)
#R_paths = get_path_list(R_dir, f_extensions=['.npy'], sub_d=True)
#W_paths = get_path_list(W_dir, f_extensions=['.npy'], sub_d=True)
#print(f'Pathy raccolti: {len(N1_paths), len(N2_paths),  len(N3_paths),  len(R_paths),  len(W_paths)}')

#dataset = SleepDataset(N1_file=N1_paths, N2_file=N2_paths, N3_file=N3_paths, R_file=R_paths, W_file=W_paths)
#print(f'Dataset creato di lunghezza: {len(dataset)}')

#train_size = int(0.8*len(dataset))
#test_size = len(dataset) - train_size
#train_ds, test_ds = random_split(dataset, [train_size, test_size])
#del dataset
#torch.save(train_ds, "/u/pmihelj/datasets/sleep/VAEEG/Train_ds.pt", _use_new_zipfile_serialization=True, pickle_protocol=5)
#del train_ds
#torch.save(test_ds, "/u/pmihelj/datasets/sleep/VAEEG/Test_ds.pt", _use_new_zipfile_serialization=True, pickle_protocol=5)
#del test_ds
#gc.collect()
#print('dataset_salvato')
BATCH_SIZE = 8
save_path = "/u/pmihelj/datasets/sleep/VAEEG/saves/ckpt_900.pt"

checkpoint = torch.load(save_path, map_location='cuda', weights_only=False)

model = SleepStage_Model().to('cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('Modello: VAEEG') ################################################################################################################################
save_dir = '/u/pmihelj/datasets/sleep/VAEEG/'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir+'/saves', exist_ok=True)
train_ds = torch.load("/u/pmihelj/datasets/sleep/VAEEG/Train_ds.pt", weights_only=False)
test_ds = torch.load( "/u/pmihelj/datasets/sleep/VAEEG/Test_ds.pt", weights_only=False)
print(f'Dataset caricato: campioni di train {len(train_ds)}, campioni di test:{len(test_ds)}')

labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(torch.int64)
class_counts = torch.bincount(labels)
class_weights = 1. / class_counts.float()
sample_weights = class_weights[labels]
tr_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

class_indices = {c: [] for c in range(5)}
for i, (_, label) in enumerate(test_ds):
    class_indices[int(label)].append(i)
te_sampler = AtLeastOnePerClassSampler(class_indices, batch_size=BATCH_SIZE)

trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, sampler=tr_sampler, pin_memory=True, num_workers=4, collate_fn=pad_collate)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, sampler=te_sampler, pin_memory=True, num_workers=4, collate_fn=pad_collate)
del train_ds, test_ds
gc.collect()
print('train loader creato')
model = SleepStage_Model()
print('Inizio training')
confmatr, accs, total_losses = train_model(model=model, n_epoch=300, trainloader=trainloader, testloader=testloader, lr=0.001, steps=100, save_dir=save_dir)
np.save(save_dir+'/conf_matr.npy', confmatr)
np.save(save_dir+'/accs.npy', accs)
np.save(save_dir+'/losses.npy', total_losses)


