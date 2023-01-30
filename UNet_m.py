# Modules for UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,  # because it passed already in the previous conv
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.bn_depth = nn.BatchNorm2d(out_channels)
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn_point = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depth(x)
        x = F.relu(x)

        x = self.pointwise(x)
        x = self.bn_point(x)
        x = F.relu(x)
        return x


"""
From https://github.com/JusperLee/AFRCNN-For-Speech-Separation
"""
class GlobalChannelLayerNorm(nn.Module):
    '''
        Global Layer Normalization
    '''
    def __init__(self, channel_size):
        super(GlobalChannelLayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)
    
    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)

    def forward(self, x):
        """
        x: N x C x T
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())
"""
Modified FFN Block of  
Li, Kai, Runxuan Yang, and Xiaolin Hu. 
"An efficient encoder-decoder architecture with top-down attention for speech separation."
arXiv preprint arXiv:2209.15200 (2022).

"""
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels) : 
        super(MultiScaleConvBlock, self).__init__()
        
        self.net = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels*2, kernel_size=1),
                    GlobalChannelLayerNorm(in_channels*2),
                    DepthwiseSeparableConv2d(in_channels*2, in_channels*2, kernel_size=1),
                    GlobalChannelLayerNorm(in_channels*2),
                    nn.Conv2d(in_channels*2, in_channels, kernel_size=1),
                    GlobalChannelLayerNorm(in_channels)
        
        )

    def forward(self,x):
        # x : [B, C, F, T]
        x = self.net(x)
        return x

class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output

## Uformer Modules
## https://arxiv.org/pdf/2111.06015.pdf

class EncoderDecoderAttention(nn.Module):
    def __init__(self,channels) -> None :
        super(EncoderDecoderAttention,self).__init__()
        
        # Encoder Kernel  
        self.w_e = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2,3),stride=1, padding="same", dilation=1)
        # Decoder Kernel  
        self.w_d = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2,3),stride=1, padding="same", dilation=1)
        # Attention Kernel  
        self.w_a = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2,3),stride=1, padding="same", dilation=1)
        
        self.sigma = nn.Sigmoid()
        
    def forward(self, d:Tensor, e:Tensor) -> Tensor:
        ## (10) extracting high dimensional feature
        g = self.sigma(self.w_e(e) + self.w_d(d))
        
        ## (11) multipling attention masking
        d_hat = torch.mul(self.sigma(self.w_a(g)),d)
        return d_hat


class OberservedAddition(nn.Module):
# Inspired by 
# Iwamoto, Kazuma, et al. "How Bad Are Artifacts?: Analyzing the Impact of Speech Enhancement Errors on ASR." arXiv preprint arXiv:2201.06685 (2022).
# https://arxiv.org/abs/2201.06685

    def __init__(self, method="none", dim=-1,factor=0.3):
        super().__init__()
        self.method = method
        self.dim = dim
        
        if method == "conv":
            self.OA = nn.Conv2d(2,1,1) 
        elif method == "linear":
            if dim == -1 :
                raise Exception("ERROR::ObversedAddtion requires 'dim' for linear layer")
            self.OA = nn.Linear(2*dim,dim)
        elif method == 'const':
            # self.factor = nn.Parameter(torch.ones(1))
            self.factor = nn.Parameter(torch.tensor(factor))
        elif method == 'fixed' : 
            self.factor = factor
        else :
            self.OA = nn.Identity()

    def forward(self, x, o):
        # x : [B C F T]
        # o : [B C F T]
        if self.method == 'conv':
            # concate on channel
            return self.OA(torch.cat((x,o),dim=1))
        elif self.method == 'linear' :
            # concate on feature
            return self.OA(torch.cat((x,o),dim=2))
        elif self.method == 'const' or self.method == 'fixed':
            return x + self.factor * o
        else :
            return self.OA(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1,norm="BatchNorm2d",padding_mode="zeros",dropout=0,activation="LeakyReLU"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        conv = nn.Conv2d

        if norm == "BatchNorm2d": 
            bn = nn.BatchNorm2d
        elif norm == "InstanceNorm2d":
            bn = nn.InstanceNorm2d
        elif norm == "LayerNorm": 
            bn = nn.LayerNorm
        else :
            raise Exception("ERROR::Encoder: unknown normalization - {}".format(norm))

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.acti = None
        if activation == "LeakyReLU":
            self.acti = nn.LeakyReLU(inplace=True)
        elif activation == "SiLU":
            self.acti = nn.SiLU(inplace=True)
        elif activation == 'Softplus':
            self.acti = nn.Softplus()
        elif activation == 'PReLU':
            self.acti = nn.PReLU()
        elif activation == 'ReLU':
            self.acti = nn.ReLU()
        elif activation == 'GELU':
            self.acti = nn.GELU()
        else :
            raise Exception("ERROR::Encoder:Unknown activation type " + str(activation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding=(0, 0),dilation=1,output_padding=(0,0),activation="LeakyReLU",norm="BatchNorm2d"):
        super().__init__()
       
        tconv = nn.ConvTranspose2d
        if norm == "BatchNorm2d": 
            bn = nn.BatchNorm2d
        elif norm == "InstanceNorm2d":
            bn = nn.InstanceNorm2d
        elif norm == "LayerNorm": 
            bn = nn.LayerNorm
        else :
            raise Exception("ERROR::Encoder: unknown normalization - {}".format(norm))
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, output_padding=output_padding,padding=padding)
        self.bn = bn(out_channels)
        self.acti = None
        if activation == "LeakyReLU":
            self.acti = nn.LeakyReLU(inplace=True)
        elif activation == "SiLU":
            self.acti = nn.SiLU(inplace=True)
        elif activation == 'Softplus':
            self.acti = nn.Softplus()
        elif activation == 'PReLU':
            self.acti = nn.PReLU()
        elif activation == 'ReLU':
            self.acti = nn.ReLU()
        elif activation == 'GELU':
            self.acti = nn.GELU()
        else :
            raise Exception("ERROR::Encoder:Unknown activation type " + str(activation))

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.acti(x)
        return x

class ResPath(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv3 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv4 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.resconv1(x)
        z1 = x1+y1
        x2 = self.conv2(z1)
        y2 = self.resconv2(z1)
        z2 = x2+y2
        x3 = self.conv3(z2)
        y3 = self.resconv3(z2)
        z3 = x3+y3
        x4 = self.conv4(z3)
        y4 = self.resconv4(z3)
        z4 = x4+y4
        return z4

"""
https://arxiv.org/abs/1608.06993v5
Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
"""
class DenseBlock(nn.Module):
    def __init__(self, n_channel,norm="BatchNorm2d",dropout=0.0,activation="LeakyReLU",depth=5):
        super().__init__()

        self.depth = depth

        ## Conv 2d
        self.seq_conv = []
        for i in range(depth):
            module = nn.conv2d(n_channel+i*n_channel,n_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1))
            seq.append(module)

        ## Normalization
        if norm == "BatchNorm2d": 
            bn_module = nn.BatchNorm2d
        elif norm == "InstanceNorm2d":
            bn_module = nn.InstanceNorm2d
        else :
            raise Exception("ERROR::Encoder: unknown nromalization - {}".format(norm))
        self.bn = bn_module(n_channel)

        if activation == "LeakyReLU":
            self.acti = nn.LeakyReLU(inplace=True)
        elif activation == "SiLU":
            self.acti = nn.SiLU(inplace=True)
        elif activation == 'Softplus':
            self.acti = nn.Softplus()
        elif activation == 'PReLU':
            self.acti = nn.PReLU()
        elif activation == 'ReLU':
            self.acti = nn.ReLU()
        elif activation == 'GELU':
            self.acti = nn.GELU()
        else :
            raise Exception("ERROR::Encoder:Unknown activation type " + str(activation))

    def forward(self,x):

        skip = []
        y = self.seq_conv[i](x)
        skip.append(y)

        for i in range(1,self.depth) : 
            y = self.seq_conv[i](torch.cat(x,skip),dim=1)
            skip.append(y)

        y0 = self.conv1(x)
        
        y0_x = torch.cat((x,y0),dim=1)
        y1 = self.conv2(y0_x)

        y1_0_x = torch.cat((x,y0,y1),dim=1)
        y2 = self.conv3(y1_0_x)

        y2_1_0_x = torch.cat((x,y0,y1,y2),dim=1)
        y3 = self.conv4(y2_1_0_x)

        y3_2_1_0_x = torch.cat((x,y0,y1,y2,y3),dim=1)
        y4 = self.conv5(y3_2_1_0_x)
        
        return y4
"""
(2022,arXiv)STFT-Domain Neural Speech Enhancement with Very Low Algorithmic Latency
https://arxiv.org/pdf/2204.09911.pdf
...
Each residual block in the encoder and decoder contains five depthwise separable 2D convolution (denoted as dsConv2D) blocks,
where the dilation rate along time are respectively 1, 2, 4, 8
and 16. Linear activation is used in the output layer to obtain
the predicted RI components.
...
"""
class ResBlock(nn.Module):
    def __init__(self, n_channel,norm="BatchNorm2d",dropout=0.0,activation="PReLU"):
        super().__init__()

        ## Normalization
        if norm == "BatchNorm2d": 
            bn= nn.BatchNorm2d
        else :
            raise Exception("ERROR::ResBlock: unknown nromalization - {}".format(norm))

        if activation == 'PReLU':
            acti = nn.PReLU()
        elif activation == 'GELU':
            self.acti = nn.GELU()
        else :
            raise Exception("ERROR::ResBlock:Unknown activation type " + str(activation))

        dilations= [
            (1,1),
            (1,2),
            (1,4),
            (1,8),
            (1,16)
        ]

        self.layers=[]
        for i in range(5):
            module = nn.Sequential(
                    nn.Conv2d(n_channel,n_channel,
                    (3,3),(1,1), # <- kernel should be (3,2)
                    dilations[i],dilations[i],n_channel),
                    nn.Conv2d(n_channel,n_channel,
                    (1,1)),
                    acti,
                    bn(n_channel)
                    )
            self.add_module("res_{}".format(i),module)
            self.layers.append(module)

    def forward(self,x):

        residual = self.layers[0](x)
        for i in range(1,5):
            residual = torch.add(residual,self.layers[i](residual))

        return residual
"""
Choi, Hyeong-Seok, et al. "Real-time denoising and dereverberation wtih tiny recurrent u-net." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.

https://github.com/YangangCao/TRUNet/blob/main/TRUNet.py
"""
class GRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, bidirectional):
        super(GRUBlock, self).__init__()
        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True, bidirectional=bidirectional)
        
        self.conv = nn.Sequential(nn.Conv1d(hidden_size * (2 if bidirectional==True else 1), out_channels, kernel_size = 1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        output,h = self.GRU(x)
        output = output.transpose(1,2)
        output = self.conv(output)
        return output

class LSTMBlock(nn.Module):
    def __init__(self,n_dim,n_hidden,n_layer=3,proj_size=None,dropout=0.2) : 
        super(LSTMBlock, self).__init__()
        
        if proj_size == None :
            proj_size = n_dim
        self.rnn = nn.LSTM(n_dim,n_hidden,n_layer,batch_first=True,proj_size=proj_size,dropout=dropout)
    
    def forward(self,x):
        # [B,C,F',T] -> [B,C*F',T]
        d0,d1,d2,d3 = x.shape
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2],x.shape[3]))
        # [B,C*F',T] -> [B,T,C*F']
        x = torch.permute(x,(0,2,1))
        #print("bottle in : {}".format(x.shape))

        x = self.rnn(x)[0]

        # [B,T,C*F'] -> [B,C*F',T]
        x = torch.permute(x,(0,2,1))
        # [B,C*F',T] -> [B,C,F',T]
        x = torch.reshape(x,(d0,d1,d2,d3))
        
        return x
    
"""
Based on routine by https://github.com/jzi040941
"""
class FGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(FGRUBlock, self).__init__()
        self.GRU = nn.GRU(
            in_channels, hidden_size, batch_first=True, bidirectional=True
        )
        # the GRU is bidirectional -> multiply hidden_size by 2
        self.conv = nn.Conv2d(hidden_size * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x):
        # X : [B, C, F', T]
        # goal : [BT, F, C ]
        B, C, T, F_ = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F_, C)
        y, h = self.GRU(x_)  # x_.shape == (BT,F,C)
        y = y.reshape(B, T, F, self.hidden_size * 2)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.conv(output)
        output = self.bn(output)
        return self.relu(output)

class TGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(TGRUBlock, self).__init__()
        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x, rnn_state=None):
        """
        X :[B, C, F', T]
        
        X' : [B*F', T, C']
        """
        B, C, F_, T = x.shape
 
        # -> [B, F', T, C]
        x = x.permute(0, 2, 3, 1)
        # -> [B*F', T, C]
        x = x.reshape(B * F_, T, C)
 
            
        x, rnn_state = self.GRU(x, rnn_state)  # y_.shape == (BF,T,C)
        #  X' : [B*F', T, hidden_size]
        # -> X' : [B, F', T, hidden_size]
        print("TGRU::{}".format(x.shape))
        x = x.reshape(B, F_, T, self.hidden_size)
        # -> X' : [B, hidden_size, F', T]
        x = x.permute(0, 3, 1, 2)     
        #  X' : [B, C, F', T]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x, rnn_state

#define custom_atan2 to support onnx conversion
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

class MEA(nn.Module):
    # class of mask estimation and applying
    def __init__(self,in_channels=4, mag_f_dim=3):
        super(MEA, self).__init__()
        self.mag_mask = nn.Conv2d(
            in_channels, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = torch.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim
    
    def forward(self, x):
        mag_mask = self.mag_mask(x)
        real_mask = self.real_mask(x).squeeze(1)
        imag_mask = self.imag_mask(x).squeeze(1)

        return (mag_mask,real_mask,imag_mask)

    def output(self,mask,feature,eps=1e-9) :
        mag_mask  = mask[0]
        real_mask = mask[1]
        imag_mask = mask[2]

        # feature [B,C,F,T]
        mag = torch.norm(feature, dim=-1)
        pha = custom_atan2(feature[..., 1], feature[..., 0])

        # stage 1
        mag_pad = F.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = F.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.relu()
        mag = mag.sum(dim=1)

        # stage 2
        mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, eps))
        pha_mask = custom_atan2(imag_mask+eps, real_mask+eps)
        real = mag * mag_mask.relu() * torch.cos(pha+pha_mask)
        imag = mag * mag_mask.relu() * torch.sin(pha+pha_mask)
        return torch.stack([real, imag], dim=-1)
    



"""
From https://github.com/JusperLee/AFRCNN-For-Speech-Separation
"""
class GlobalChannelLayerNorm(nn.Module):
    '''
        Global Layer Normalization
    '''
    def __init__(self, channel_size):
        super(GlobalChannelLayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)
    
    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)

    def forward(self, x):
        """
        x: N x C x T
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())
"""
Modified FFN Block of  
Li, Kai, Runxuan Yang, and Xiaolin Hu. 
"An efficient encoder-decoder architecture with top-down attention for speech separation."
arXiv preprint arXiv:2209.15200 (2022).

"""
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels) : 
        super(MultiScaleConvBlock, self).__init__()
        
        self.net = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels*2, kernel_size=1),
                    GlobalChannelLayerNorm(in_channels*2),
                    DepthwiseSeparableConv2d(in_channels*2, in_channels*2, kernel_size=1),
                    GlobalChannelLayerNorm(in_channels*2),
                    nn.Conv2d(in_channels*2, in_channels, kernel_size=1),
                    GlobalChannelLayerNorm(in_channels)
        
        )

    def forward(self,x):
        # x : [B, C, F, T]
        x = self.net(x)
        return x