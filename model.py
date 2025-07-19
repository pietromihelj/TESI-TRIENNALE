import torch
import torch.nn as nn
from torch.autograd import Variable

class C1dLayer(nn.Module):
    #la classe implementa semplicemente un padding certo per avere l'output  
    #della stessa lunghezza dell'input
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, bias=True):
        super(C1dLayer, self).__init__()

        #padding necessario totale
        total_p = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        #padding destro
        left_p = total_p // 2
        #padding sinistro
        right_p = total_p - left_p

        self.conv = nn.Sequential(nn.ConstantPad1d((left_p, right_p), 0),
                                  nn.Conv1d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride, dilation=dilation,
                                            bias=bias))

    def forward(self, x):
        return self.conv(x)

class CT1dLayer(nn.Module):
    #come sopra ma il layer per il decoder
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, bias=True):
        super(CT1dLayer, self).__init__()
        
        #calcolo del padding
        p = (dilation * (kernel_size - 1)) // 2
        op = stride - 1

        self.fconv = nn.ConvTranspose1d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=p,
                                        output_padding=op,
                                        dilation=dilation, bias=bias)

    def forward(self, x):
        return self.fconv(x)
    
class LSTMLayer(nn.Module):
    #semplicemente butto via lo stato interno che non mi serve
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.0, bidirectional=False):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs

class 