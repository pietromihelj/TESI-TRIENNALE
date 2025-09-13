import torch 
import torch.nn as nn
import mne
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, random_split
from utils import get_path_list, stride_data
import numpy as np
import gc
from tqdm import tqdm

SAVE_FILE = "./SubAnalyze_SeizureDetection/Model/Models"

"""
Nella parte di codice seguente vado a creare il dato in forma (clip,label), il dataset ed il dataloader. poi alleno e testoun mdodello NN
misto linear LSTM. 
"""
def load_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data = raw.get_data()  
    data_concat = data.flatten()
    return data_concat

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
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for name, parma in m.named_parameters():
                if name.find('weight') != -1 :
                    nn.init.xavier_normal_(parma)
                elif name.find('bias_ih') != -1:
                    nn.init.constant_(parma, 2)
    return model

class EEG_SEIZ_Dataset(Dataset):
    def __init__(self, norm_paths, seiz_paths, n_per_seg=250, n_overlap=0):
        self.samples = []
        self.labels = []

        for path in norm_paths:
            data = load_edf(path)
            clips = stride_data(data,n_per_seg=n_per_seg, n_overlap=n_overlap)
            self.samples.append(clips)
            self.labels.append(np.zeros(len(clips)))
        
        for path in seiz_paths:
            data = load_edf(path)
            clips = stride_data(data,n_per_seg=n_per_seg, n_overlap=n_overlap)
            self.samples.append(clips)
            self.labels.append(np.ones(len(clips)))
        
        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

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

class SeizureDetect_Model(nn.Module):

    def __init__(self, target_channels=8):
        super(SeizureDetect_Model, self).__init__()
        self.flatten0 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.headlayer = HeadLayer(19, target_channels)
        # reshape (8, 50) than transpose (10, 8, 50) than flatten 
        self.flatten1 = nn.Flatten(start_dim=-2, end_dim=-1)

        self.lstm0 = nn.LSTM(50 * target_channels, 256, num_layers=2, batch_first=True,  bidirectional=False, dropout=0.1)
        self.lstm1 = nn.LSTM(256, 16, num_layers=1, batch_first=True,  bidirectional=False)

        self.linear = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1), nn.Linear(10 * 16, 32), nn.ReLU(), nn.Dropout(p=0.1),
                                    nn.Linear(32, 1), nn.Sigmoid())
        
    def forward(self, x):
        x = self.flatten0(x)
        x = self.headlayer(x)
        x = torch.split(x, 50, dim=-1)
        x = torch.stack(x, dim=1)
        x = self.flatten1(x)
        x, _ = self.lstm0(x)
        x, _ = self.lstm1(x)
        res = self.linear(x)
        return res 
    
def bce_Loss(input, target, weight=None, device='gpu'):
    if weight:
        weight_ = torch.ones(target.size()).to(device)
        weight_[target==0] = weight
        return nn.BCELoss(weight=weight_)(input, target)
    else:
        return nn.BCELoss()(input, target)

def save_model(model, file):
    torch.save(model.state_dict(), file)

def load_model(model_frame, file):
    model_frame.load_state_dict(torch.load(file))
    return model_frame

def get_metrics(predict, label, class_num=2):
    predict = predict.round()
    metric_cm = metrics.confusion_matrix(label, predict, labels=list(range(class_num)))
    metric = metric_cm / metric_cm.sum(axis=1, keepdims=True)
    acc0, acc1 = np.diagonal(metric)
    acc_total = metrics.accuracy_score(label, predict)
    return metric_cm, np.round(np.array([acc0, acc1, acc_total]), 3)

def train_model(model, n_epoch, trainloader, testloader, lr, device='cuda', weigth = 0.15):

    model = SeizureDetect_Model().to(device)
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_losses = []
    for e in range(n_epoch):
        model.train()
        tr_losses = []
        tr_accs = []

        for tr_data, labels in trainloader:
            pred = model(tr_data.float.to(device))
            loss = bce_Loss(pred, labels.float().to(device), weight=weigth, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss)
            _, accs = get_metrics(pred.to("cpu").detach().numpy(), labels.to("cpu").detach().numpy(), class_num=2)
            tr_accs.append(accs[2])
        total_losses.append(np.mean(tr_losses))
        print(f'Epoch: {e}, Loss: {np.mean(tr_losses)}, Accuracy: {np.mean(tr_accs)}')
    
    model.eval()
    te_CMs = []
    accs = []
    rec_seiz = []
    rec_norm = []
    for te_data, labels in testloader:
        te_pred = model(te_data.float().to(device))
        te_cm, te_accs = get_metrics(te_pred.to('cpu').detach().numpy(), labels.to('cpu').detach().numpy(), class_num=2)
        te_CMs.append(te_cm)
        accs.append(te_accs[2])
        rec_norm(te_accs[0])
        rec_seiz(te_accs[1])

    print(f'Test accuracy:{torch.tensor(accs).mean()}, recall norm: {torch.tensor(rec_norm).mean()}, recall seizure: {torch.tensor(rec_seiz).mean()}')
    print(f'Confuzion Metrics: {torch.tensor(te_CMs).mean(dim=0)}')

    return np.mean(te_CMs, axis=0), np.mean(accs), np.mean(rec_norm), np.mean(rec_seiz), total_losses


normal_dir = "D:/seizure_dataset/normal"
seizure_dir = "D:/seizure_dataset/seizure"

normal_paths = get_path_list(normal_dir, f_extensions=['.edf'], sub_d=True)
seizure_paths = get_path_list(seizure_dir, f_extensions=['.edf'], sub_d=True)

dataset = EEG_SEIZ_Dataset(norm_paths=normal_paths, seiz_paths=seizure_paths, n_per_seg=50)

train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
torch.save(train_ds, "D:/Train_seizure_dataset.pt")
torch.save(test_ds, 'D:/Test_seizure_dataset.pt')
trainloader = DataLoader(train_ds, batch_size=32, shuffle=True)
testloader = DataLoader(test_ds, batch_size=32, shuffle=False)
model = SeizureDetect_Model()
confmatr, accs, recall_norm, recall_seiz = train_model(model, 100, trainloader, testloader, 0.001)
np.save('seiz_detec/conf_matr.npy', confmatr)
np.save('seiz_detec/accs.npy', accs)
np.save('seiz_detec/recall_norm.npy', recall_norm)
np.save('seiz_detec/recall_seiz.npy', recall_seiz)

"""
Nella seguente parte di codice paragono le distribuzioni latenti tra background e seizure
"""

normal_dir = "D:/seizure_dataset/normal"
seizure_dir = "D:/seizure_dataset/seizure"

normal_paths = get_path_list(normal_dir, f_extensions=['.edf'], sub_d=True)
seizure_paths = get_path_list(seizure_dir, f_extensions=['.edf'], sub_d=True)

clips_norm = []
print('Normal Processing')
for path in tqdm(normal_paths):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    data = raw.get_data()  
    clip = stride_data(data,n_per_seg=50)
    clips_norm.append(np.squeeze(clip).mean(axis=0))
    del raw, data, clip
    gc.collect()
clips_norm = np.concatenate(clips_norm, axis=0)

print('Seizure Processing')
clips_seiz = []
for path in tqdm(seizure_paths):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    data = raw.get_data()  
    clip = stride_data(data,n_per_seg=50)
    clips_seiz.append(np.squeeze(clip).mean(axis=0))
    del raw, data, clip
    gc.collect()
clips_seiz = np.concatenate(clips_seiz, axis=0)


print('Distribution comparison')
distributions = {}
for i in range(50):
    latent_norm = clips_norm[:, i]
    latent_seiz = clips_seiz[:, i]
    distributions[f'latent_{i}'] = {'normal': latent_norm, 'seizure': latent_seiz}

from scipy.stats import mannwhitneyu
p_values = []
for i in range(50):
    norm_vals = distributions[f'latent_{i}']['normal']
    seiz_vals = distributions[f'latent_{i}']['seizure']
    stat, p = mannwhitneyu(norm_vals, seiz_vals)
    print(f"Latent {i}: p-value={p}")
    p_values.append(p)
np.save('seiz_detec/p_values.npy', p_values)

import matplotlib.pyplot as plt
print('Distribution plotting')
n_latent = 50
n_rows, n_cols = 5, 10  
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10)) 
for i in range(n_latent):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    ax.hist(distributions[f'latent_{i}']['normal'], bins=30, alpha=0.5, label='normal')
    ax.hist(distributions[f'latent_{i}']['seizure'], bins=30, alpha=0.5, label='seizure')
    ax.set_title(f'Latent {i}', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

plt.savefig("seiz_detec/latent_distributions.png", dpi=300)
plt.tight_layout()
plt.show()






