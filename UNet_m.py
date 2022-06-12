# Modules for UNet
import torch
import torch.nn as nn
from torch import Tensor

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, padding_mode="zeros",dropout=0,activation="LeakyReLU"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        conv = nn.Conv2d
        bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding,padding=(0, 0),activation="LeakyReLU"):
        super().__init__()
       
        tconv = nn.ConvTranspose2d
        bn = nn.BatchNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=output_padding,padding=padding)
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