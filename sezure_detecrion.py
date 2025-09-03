import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

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

class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, bias=True):
        super(Conv1dLayer, self).__init__()
        total_p = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        left_p = total_p // 2
        right_p = total_p - left_p
        self.conv = nn.Sequential(nn.ConstantPad1d((left_p, right_p), 0), nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias))
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

        self.conv1 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit, kernel_size=9, stride=1, bias=False), nn.BatchNorm1d(unit), nn.LeakyReLU(negative_slope))
        self.conv2 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit, kernel_size=7, stride=1, bias=False), nn.BatchNorm1d(unit), nn.LeakyReLU(negative_slope))
        self.conv3 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit, kernel_size=5, stride=1, bias=False), nn.BatchNorm1d(unit), nn.LeakyReLU(negative_slope))
        self.conv4 = nn.Sequential(Conv1dLayer(in_channels=in_channels, out_channels=unit, kernel_size=3, stride=1, bias=False), nn.BatchNorm1d(unit), nn.LeakyReLU(negative_slope))
        self.conv5 = nn.Sequential(Conv1dLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, bias=False), nn.BatchNorm1d(out_channels), nn.LeakyReLU(negative_slope))

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
        self.linear = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1), nn.Linear(10 * 16, 32), nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(32, 1), nn.Sigmoid())
        
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
    
class dataset_sezure(Dataset):
    def __init__(self, model):
        
