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

        self.conv = nn.Sequential(nn.ConstantPad1d((left_p, right_p), 0), nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias))

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

        self.convT = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=p, output_padding=op, dilation=dilation, bias=bias)

    def forward(self, x):
        return self.convT(x)
    
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

class HeadLayer(nn.Module):
    #ricavo una serie di feature da immettere nella rete con kernel di dimensioni diverse
    #per gestire vari livelli di generalizzazione contemporaneamente
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(HeadLayer, self).__init__()

        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4, but got: %d" % out_channels)

        unit = out_channels // 4

        self.conv1 = nn.Sequential(C1dLayer(in_channels=in_channels, out_channels=unit, kernel_size=11, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv2 = nn.Sequential(C1dLayer(in_channels=in_channels, out_channels=unit, kernel_size=9, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv3 = nn.Sequential(C1dLayer(in_channels=in_channels, out_channels=unit, kernel_size=7, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv4 = nn.Sequential(C1dLayer(in_channels=in_channels, out_channels=unit, kernel_size=5, stride=1, bias=False),
                                   nn.BatchNorm1d(unit),
                                   nn.LeakyReLU(negative_slope))

        self.conv5 = nn.Sequential(C1dLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm1d(out_channels),
                                   nn.LeakyReLU(negative_slope))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        #unisco i 4 dati risultanti lungo la dimensione delle feature per incorporare quelle
        #estratte dalle varie generalizzazioni
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv5(out)
        return out
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, negative_slope=0.2):
        super(ResBlock, self).__init__()

        #se il numero di canali di input è uguale all'ouput e lostride è 1, cioè il dato non 
        #cambia dimensione allora faccio solo le 2 convoluzioni e attivazione
        if stride == 1 and in_channels == out_channels:
            self.projection = None
        else:
        #in caso contrario proietto i dati sulla giusta dimensione di output
            self.projection = nn.Sequential(C1dLayer(in_channels, out_channels, 1, stride, bias=False),
                                            nn.BatchNorm1d(out_channels))

        self.conv1 = nn.Sequential(C1dLayer(in_channels, out_channels, kernel_size, stride, bias=False),
                                nn.BatchNorm1d(out_channels),
                                nn.LeakyReLU(negative_slope))

        self.conv2 = nn.Sequential(C1dLayer(out_channels, out_channels, kernel_size, 1, bias=False),
                                nn.BatchNorm1d(out_channels))

        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        if self.projection:
            res = self.projection(x)
        else:
            res = x

        out = self.conv1(x)
        out = self.conv2(out)
        #sommo il residuo per evitare gradient vanishing or explsion nel training migliorando
        #la stabilità
        out = out + res
        out = self.act(out)
        return out
    
def re_parameterize(mu, log_var):
    #formulazione standard del reparametrization trick per la normale
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

def z_sample(z):
    zp = torch.randn_like(z, requires_grad=True)
    return zp


def recon_loss(x, x_bar):
    #loss di ricostruzione come differenza tra input e output
    value = torch.nn.functional.mse_loss(x, x_bar, reduction="mean")
    return value

def kl_loss(mu, log_var):
    #divergenza di Kullback-Leibler come regolarizzazione nella forma chiusa per la prior normale
    value = torch.mean(-0.5 * (1 + log_var - torch.exp(log_var) - mu ** 2))
    return value

class Encoder(nn.Module):
    def __init__(self, in_channels, z_dim, negative_slope=0.2):
        super(Encoder,self).__init__()
        #metto il layer d'entrata che estrae delle feature con diverse generalizzazioni del dato
        self.layers = nn.ModuleList([HeadLayer(in_channels=in_channels,out_channels=16,negative_slope=negative_slope)])

        #setto il numero di features, corrispondente al numero di canali
        in_features = [16,16,24,32]
        out_features = [16,24,32,32]
        n_rblocks = [2,2,2,2]

        for in_ch, out_ch, n_rbloc in zip(in_features, out_features, n_rblocks):
            self.layers.append(nn.Sequential(C1dLayer(in_ch,out_ch,3,2,bias=False),
                                             nn.BatchNorm1d(out_ch),
                                             nn.LeakyReLU(negative_slope)))
            for _ in range(n_rbloc):
                self.layers.append(ResBlock(out_ch,out_ch,3,1,negative_slope))
        self.layers.append(nn.Sequential(nn.Flatten(1),
                                         nn.Linear(250,32),
                                         nn.BatchNorm1d(32),
                                         nn.LeakyReLU(negative_slope)))
        self.mu = nn.linear(32, z_dim)
        self.log_var = nn.Linear(32,z_dim)
    
    def forward(self, x):
        for L in self.layers:
            x = L(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, z_dim, negative_slope=0.2, last_lstm=True):
        super(Decoder,self).__init__()
        self.conT = nn.Sequential(nn.Linear(z_dim,250),
                                  nn.BatchNorm1d(250),
                                  nn.LeakyReLU(negative_slope))
        in_features = [32, 32, 24, 16, 16]
        out_features = [32, 24, 16, 16, 8]
        n_rblocks = [2, 2, 2, 2, 2]
        self.layers = nn.ModuleList()

        for in_ch, out_ch, n_rblock in zip(in_features, out_features, n_rblocks):
            self.layers.append(nn.Sequential(CT1dLayer(in_ch, out_ch, 3, 2, bias=False),
                                             nn.BatchNorm1d(out_ch),
                                             nn.LeakyReLU(negative_slope)))
            for _ in range(n_rblock):
                self.layers.append(ResBlock(out_ch, out_ch, 3, 1, negative_slope))
        
        self.layers.append(nn.Sequential(C1dLayer(out_features[-1],out_features[-1], 3, 1, bias=False),
                                         nn.BatchNorm1d(out_features[-1]),
                                         nn.LeakyReLU(negative_slope)))
        
        if last_lstm:
            self.tail = LSTMLayer(out_features[-1], 1, 2)
        else:
            self.tail = nn.Sequential(C1dLayer(out_features[-1], out_features[-1] // 2, 5, 1, bias=True),
                                      nn.BatchNorm1d(out_features[-1] // 2),
                                      nn.LeakyReLU(negative_slope),
                                      C1dLayer(out_features[-1] // 2, 1, 3, 1, bias=True))
        self.last_lstm = last_lstm

    def forward(self, x):
        x = self.conT(x)
        n_batch, nf = x.shape
        x = x.view(n_batch, 32, 8)

        for L in self.layers:
            x = L(x)

        if self.last_lstm:
            x = torch.permute(x,(2,0,1))
            x = self.tail(x)
            x = torch.permute(x,(1,2,0))
        else:
            x = self.tail(x)
        return x

class VAEEG(nn.Module):
    def __init__(self, in_channels, z_dim, negative_slope=0.2, decoder_last_lstm=True):
        super(VAEEG,self).__init__()
        self.encoder = Encoder(in_channels=in_channels, z_dim=z_dim, negative_slope=negative_slope)
        self.decoder = Decoder(z_dim=z_dim, negative_slope=negative_slope, last_lstm=decoder_last_lstm)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = re_parameterize(mu, log_var)
        x_rec = self.decoder(z)
        return mu, log_var, x_rec