"""
Directional Separating UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attractor(nn.Module) :
    def __init__(self,n_ch=4,n_fft=512):
        super(Attractor,self).__init__()

        self.enc_SV = nn.Linear(n_ch *(n_fft+2), n_fft//2 +1)
        self.act_SV = nn.PReLU()

        self.enc_theta = nn.Linear(2,n_fft//2 +1)
        self.act_theta = nn.Sigmoid()

        self.bn = nn.BatchNorm1d(n_fft//2 +1)

        self.enc_2 = nn.Linear(n_fft//2 +1, n_fft+2)
        self.act_2 = nn.PReLU()
        self.enc_3 = nn.Linear(n_fft+2, n_fft//2)
        self.act_3 = nn.Sigmoid()

    def forward(self, SV, theta):
        SV = torch.reshape(SV,(SV.shape[0],-1))
        a = self.enc_SV(SV)
        a = self.act_SV(a)

        b = self.enc_theta(theta)
        b = self.act_theta(b)

        attract = a + b
        attract = self.bn(attract)
        attract = self.enc_2(attract)
        attract = self.act_2(attract)
        attract = self.enc_3(attract)
        attract = self.act_3(attract)

        return attract
    
class AttractEncoder(nn.Module):
    def __init__(self,n_ch, n_fft = 512):
        super(AttractEncoder,self).__init__()
        self.n_ch = n_ch

        self.enc= nn.Linear(n_fft//2,n_ch)
        self.conv = nn.Conv1d(1,n_ch,kernel_size=1)
        self.act = nn.Sigmoid()

    # attract : [B,F]
    def forward(self,attract):
        attract = self.enc(attract)
        # attract : [B, n_ch]
        attract = self.act(attract)
        attract = torch.clamp(attract,min=1/self.n_ch)
        return attract

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, padding_mode="zeros",activation="LeakyReLU",dropout=0.0,type_norm = "BatchNorm2d"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
       
        conv = nn.Conv2d
        if type_norm == "BatchNorm2d" : 
            bn = nn.BatchNorm2d
        elif type_norm == "InstanceNorm2d" :
            bn = nn.InstanceNorm2d

        self.dr = nn.Dropout(dropout)

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)

        if activation == "LeakyReLU" : 
            self.relu = nn.LeakyReLU(inplace=True)
        elif activation == "PReLU" :
            self.relu = nn.PReLU()
        elif activation == "SiLU" :
            self.relu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dr(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding,padding=(0, 0), activation = "LeakyReLU",type_norm = "BatchNorm2d"):
        super().__init__()
        
        tconv = nn.ConvTranspose2d
        if type_norm == "BatchNorm2d" : 
            bn = nn.BatchNorm2d
        elif type_norm == "InstanceNorm2d" :
            bn = nn.InstanceNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=output_padding,padding=padding)
        self.bn = bn(out_channels)

        if activation == "LeakyReLU" : 
            self.relu = nn.LeakyReLU(inplace=True)
        elif activation == "PReLU" :
            self.relu = nn.PReLU()
        elif activation == "SiLU" :
            self.relu = nn.SiLU()

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class TGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, skipGRU=False,**kwargs):
        super(TGRUBlock, self).__init__()

        if not skipGRU : 
            self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True)
        else : 
            raise Exception("Not Implemented")
            #self.GRU = SkipGRU(in_channels*2, hidden_size*2, batch_first=True)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    # x : torch.Size([B, C=128, F=2, T=16, RI=2])
    def forward(self, x, rnn_state=None):
        B, C, F, T = x.shape  # x.shape == (B, C, T, F)

        # unpack, permute, and repack
        x1 = x.permute(0, 2, 3, 1)  # x2.shape == (B,F,T,C,2)
        x_ = x1.reshape(B * F, T, C)  # x_.shape == (BF,T,C*2)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C*2)
        # unpack, permute, and repack
        y1 = y_.reshape(B, F, T, self.hidden_size)  # y1.shape == (B,F,T,C,2)
        y2 = y1.permute(0, 3, 1, 2)  # y2.shape == (B,C,F,T,2)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output
    
# omplex Ratio Mask
class CRM(nn.Module):
    def __init__(self):
        super(CRM,self).__init__()
    
    def forward(self,x):
        mask = torch.tanh(x)
        mask = mask[:,0,...] + 1j*mask[:,1,...]

        return mask
    
    def output(self,X,M):
        return X*M

# Complex Mapping
class CM(nn.Module):
    def __init__(self):
        super(CM,self).__init__()

    def forward(self,x):
        map= x[:,0,...] + 1j*x[:,1,...]
        return map
    
    def output(self,X,M):
        return M


# Complex Mask Estimation and Applying    
class CMEA(nn.Module):
    # class of mask estimation and applying
    def __init__(self,in_channels=1, mag_f_dim=3):
        super(CMEA, self).__init__()
        self.mag_mask = nn.Conv2d(
            in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.mag_f_dim = mag_f_dim

    #define custom_atan2 to support onnx conversion
    @staticmethod
    def custom_atan2(y, x):
        pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
        ans = torch.atan(y / (x + 1e-6))
        ans += ((y > 0) & (x < 0)) * pi
        ans -= ((y < 0) & (x < 0)) * pi
        ans *= 1 - ((y > 0) & (x == 0)) * 1.0
        ans += ((y > 0) & (x == 0)) * (pi / 2)
        ans *= 1 - ((y < 0) & (x == 0)) * 1.0
        ans += ((y < 0) & (x == 0)) * (-pi / 2)
        return ans
    
    # x : [B,C,F,T,2]
    def forward(self, x):
        mag_mask = self.mag_mask(torch.sqrt(x[...,0]**2+x[...,1]**2)).squeeze(1)
        real_mask = self.real_mask(x[...,0]).squeeze(1)
        imag_mask = self.imag_mask(x[...,1]).squeeze(1)

        return (mag_mask,real_mask,imag_mask)

    def output(self,feature,mask,eps=1e-9) :
        mag_mask  = mask[0]
        real_mask = mask[1]
        imag_mask = mask[2]

        # feature [B,C,F,T]
        mag = torch.abs(feature)
        pha = self.custom_atan2(feature.imag, feature.real)

        # stage 1
        # v83
        #mag = mag * F.softplus(mag_mask)
        # v84
        mag = mag * F.sigmoid(mag_mask)

        # stage 2
        mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, eps))
        pha_mask = self.custom_atan2(imag_mask+eps, real_mask+eps)

        # default
        #real = mag * mag_mask.relu() * torch.cos(pha+pha_mask)
        #imag = mag * mag_mask.relu() * torch.sin(pha+pha_mask)

        # v85 sigmoid
        real = mag * mag_mask.sigmoid() * torch.cos(pha+pha_mask)
        imag = mag * mag_mask.sigmoid() * torch.sin(pha+pha_mask)
        return torch.stack([real, imag], dim=-1)



class DSUNet(nn.Module):
    def __init__(self, 
                 input_channels=8,
                 n_fft=512,
                 #model_complexity=45,
                 model_complexity=45,
                 bottleneck="None",
                 padding_mode="zeros",
                 type_masking = "CRM",
                 activation = "LeakyReLU",
                 dropout=0.0,
                 type_norm = "BatcNorm2d"
                 ):
        super().__init__()

        self.bottleneck = bottleneck

        self.n_channel = input_channels
        n_angle = 2

        print("DSUNet::complexity {}".format(model_complexity))

        model_depth=20

        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.model_length = model_depth // 2
        self.dropout = dropout

        ## Encoder
        self.encoders = []
        module_cls = Encoder

        for i in range(self.model_length):
            module = module_cls(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],stride=self.enc_strides[i], padding=self.enc_paddings[i], padding_mode=padding_mode,dropout = dropout,activation=activation,type_norm = type_norm)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        ## Decoder
        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], output_padding=self.dec_output_paddings[i],
                             activation=activation
                             )
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        # Bottleneck
        if bottleneck == "TGRU" : 
            self.BTN = TGRUBlock(128,128,128)
        else :
            self.BTN = nn.Identity()
            
        ## Attractor
        self.Attractor = Attractor(n_ch=4,n_fft=n_fft)

        self.attractEncoders = [] 
        for i in range(self.model_length) : 
            module = AttractEncoder(
                n_ch= self.enc_channels[i+1]
            )
            self.add_module("attractEncoder_{}".format(i),module)
            self.attractEncoders.append(module)


        conv = nn.Conv2d
        linear = conv(self.dec_channels[-1], 2, 1)

        ## Mask Estimator
        self.add_module("linear", linear)
        self.padding_mode = padding_mode

        if type_masking == "CRM" : 
            self.mask = CRM()
        elif type_masking == "CMEA" : 
            self.mask = CMEA()
        elif type_masking == "CM" : 
            self.mask = CM()
        else :
            raise Exception("Not Implemented")

        self.dr = nn.Dropout(self.dropout)

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)
        self.attractors = nn.ModuleList(self.attractEncoders)

    def forward(self, sf,SV,theta):        
        # ipnut : [ Batch Channel Freq Time 2]

        attract = self.Attractor(SV,theta)
        #print("attract : {}".format(attract.shape))

        # Encoders
        sf_skip = []
        for i, encoder in enumerate(self.encoders):
            sf_skip.append(sf)
            sf = encoder(sf)
            sf = self.dr(sf)
            #print("sf{}".format(i), sf.shape)

            a_s = self.attractEncoders[i](attract)
            a_s = torch.reshape(a_s,(*a_s.shape,1,1))
            #print("a_s{} : {}".format(i, a_s.shape))
            sf = a_s*sf
        # sf_skip : sf0=input sf1 ... sf9

        #print("fully encoded ",sf.shape)
        p = sf

        # Bottleneck

        if self.bottleneck == "RNN_AS" or self.bottleneck == "AATT": 
            p = self.BTN(p,attract)
        else : 
            p = self.BTN(p)

        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            #print(f"p{i}, {p.shape} + sf{self.model_length - 1 - i}, {sf_skip[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            
            p = torch.cat([p, sf_skip[self.model_length - 1 - i]], dim=1)

        #print(p.shape)
        p = self.linear(p)
        # p : [B,1,F,T,2]
        M = self.mask(p)
        return M
    
    def masking(self,X,M):
        return self.mask.output(X,M)

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        self.enc_channels = [input_channels,
                                model_complexity,
                                model_complexity,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                128]

        self.enc_kernel_sizes = [(7, 1),
                                    (1, 7),
                                    (7, 5),
                                    (7, 5),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3)]

        self.enc_strides = [(1, 1),
                            (1, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1)]

        self.enc_paddings = [(3, 0),
                                (0, 3),
                                (3, 2),
                                (3, 2),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),]
                            
        self.dec_channels = [0,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2]

        self.dec_kernel_sizes = [(5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3), 
                                    (7, 5), 
                                    (7, 5), 
                                    (1, 7),
                                    (7, 1)]

        self.dec_strides = [(2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (1, 1),
                            (1, 1)]

        self.dec_paddings = [(2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (3, 2),
                                (3, 2),
                                (0, 3),
                                (3, 0)]
        
        self.dec_output_paddings = [(0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,0)]
        
class DSUNet_helper(nn.Module):
    def __init__(self,
                 n_fft = 512,
                 dropout = 0.0,
                 bottleneck = "None",
                 model_complexity = 45,
                 use_SV = True,
                 mag_phase = False,
                 corr = False,
                 DSB = False,
                 activation= "LeakyReLU",
                 type_norm ="BatchNorm2d",
                 type_masking="CRM"
                 ):
        super(DSUNet_helper,self).__init__()

        self.n_fft = n_fft
        self.n_hfft = n_fft // 2 + 1

        self.n_channel = 4

        model_channel = self.n_channel*2

        if DSB :
            model_channel += 1

        self.mag_phase = mag_phase

        self.use_SV = use_SV
        self.corr = corr 
        self.DSB = DSB

        self.net = DSUNet(input_channels = model_channel,
                        dropout = dropout,
                        bottleneck=bottleneck,
                        model_complexity=model_complexity,
                        activation=activation,
                        type_norm = type_norm,
                        type_masking=type_masking
                        )
        
        # const
        self.sr = 16000
        self.ss = 340.4
        self.dist = 100

        self.window = torch.hann_window(self.n_fft)

    def forward(self,x,angle,mic_pos):
        B,C,L = x.shape

        x = torch.reshape(x,(B*C,L))
        X = torch.stft(x,n_fft = self.n_fft,window=self.window.to(x.device),return_complex=True)
        _, F,T = X.shape
        short = 16-T%16
        X = torch.nn.functional.pad(X,(0,short))
        _, F,T = X.shape
        X = X.reshape(B,C,F,T)

        if self.corr : 
            for i in range(B) : 
                for j in range(C-1) : 
                    X[i,j+1] = X[i,0] * X[i,j+1]

        SV = self.steering_vector(angle,mic_pos)
        SV = torch.cat((SV.real,SV.imag),-1)

        theta = self.anlge_pre(angle)
        #print("angle feaute : {}".format(angle_feature.shape))

        # [B,C,F,T,2]
        spectral_feature = torch.cat((X.real,X.imag),dim=1)

        if self.DSB : 
            # B,F,T
            DSB = self.delay_n_sum(X,angle,mic_pos)
            # B,F,T,2
            DSB = torch.stack((DSB.real,DSB.imag),dim=-1)
            # B,1,F,T,2
            DSB = torch.unsqueeze(DSB,1)

            # B,C+1,F,T,2
            spectral_feature = torch.cat((spectral_feature,DSB),dim=1)

        mask = self.net(spectral_feature,SV,theta)

        #print("mask {} | X {}".format(mask.shape,X.shape))

        Y = self.net.masking(X[:,0],mask)
        y = torch.istft(Y,n_fft = self.n_fft,window=self.window.to(Y.device),length=L)
        return y

    def anlge_pre(self,angle):
        sin_theta = torch.sin((-angle)/180*torch.pi)
        cos_theta = torch.cos((-angle)/180*torch.pi)
        return torch.stack((sin_theta,cos_theta),1)
    
    def steering_vector(self,angle,mic_pos) :
        """
        angle : [B]
        """
        # init
        B = angle.shape[0]
        loc_src = torch.zeros(B,3).to(mic_pos.device)

        loc_src[:,0] = self.dist*torch.cos((-angle)/180*torch.pi)*torch.sin(torch.tensor(90/180*torch.pi))
        loc_src[:,1] = self.dist*torch.sin((-angle)/180*torch.pi)*torch.sin(torch.tensor(90/180*torch.pi))
        loc_src[:,2] = self.dist*torch.cos(torch.tensor((90)/180*torch.pi))

        TDOA = torch.zeros(B,self.n_channel).to(mic_pos.device)
        for i in range(self.n_channel) : 
            TDOA[:,i] = torch.norm(mic_pos[:,i] - loc_src)
        TDOA = TDOA[:,:] - TDOA[:,0:1]

        const = -1j*2*torch.pi*self.sr/(self.n_fft*self.ss)
        SV  = torch.zeros(B,self.n_channel,self.n_hfft,dtype=torch.cfloat).to(mic_pos.device)
        for i in torch.arange(B) : 
            for j in range(self.n_hfft) : 
                SV[i,:,j] = torch.exp(j*TDOA[i]*const)
                SV[i,:,j] /= torch.norm(SV[i,:,j])

        return SV
    
    # Delay-and-Sum Beamformer
    def delay_n_sum(self,X,angle,mic_pos):
        """
        angle : [B]
        """
        # init
        B = angle.shape[0]
        loc_src = torch.zeros(B,3).to(mic_pos.device)

        loc_src[:,0] = self.dist*torch.cos((-angle)/180*torch.pi)*torch.sin(torch.tensor(90/180*torch.pi))
        loc_src[:,1] = self.dist*torch.sin((-angle)/180*torch.pi)*torch.sin(torch.tensor(90/180*torch.pi))
        loc_src[:,2] = self.dist*torch.cos(torch.tensor((90)/180*torch.pi))

        TDOA = torch.zeros(B,self.n_channel).to(mic_pos.device)
        for i in range(self.n_channel) : 
            TDOA[:,i] = torch.norm(mic_pos[:,i] - loc_src)
        TDOA = TDOA[:,:] - TDOA[:,0:1]

        const = 1j*2*torch.pi*self.sr/(self.n_fft*self.ss)
        h  = torch.zeros(B,self.n_channel,self.n_hfft,dtype=torch.cfloat).to(mic_pos.device)
        for i in torch.arange(B) : 
            for j in range(self.n_hfft) : 
                h[i,:,j] = torch.exp(j*TDOA[i]*const)
                h[i,:,j] /= torch.norm(h[i,:,j])

        h = torch.unsqueeze(h,-1)

        Y = torch.einsum('bcft,bcfl->bft', [X, h])
        return Y

def test() : 
    B = 2
    C = 4
    F = 257
    L = 32000
    T = 256

    x = torch.rand(B,C,L)
    angle = torch.rand(B)
    mic_pos = torch.rand(B,4,3)

    m = DSUNet_helper(type_masking="CRM")

    y = m(x,angle,mic_pos)

    print("output : {}".format(y.shape))