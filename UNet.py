# Tensorflow code -> Pytorch Code 
# of vblez/Speech-enhancement
# https://github.com/vbelz/Speech-enhancement/blob/master/model_unet.py
from shutil import ExecError
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
from torch import Tensor

from .UNet_m import Encoder, Decoder, EncoderDecoderAttention, ResPath, OberservedAddition

class UNet(nn.Module):
    def __init__(self, 
                 input_channels=1,
                 dropout=0.2,
                 activation="PReLU",
                 mask_activation="Softplus",
                 bottleneck_type="None",
                 n_fft=512,
                 res_path="None",
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros",
                 use_EDA=False, # Encoder Decoder Attention
                 OA_method = "none",
                 OA_dim = 257,
                 OA_factor=0.3,
                 device="cuda:0"
                 ):
        super().__init__()

        self.nhfft = n_fft/2 + 1
        self.input_channels = input_channels

        self.model_complexity = int(model_complexity // 1.414)

        self.encoders = []
        self.model_length = model_depth // 2

        self.set_size(model_depth)

        self.device =device


        if use_EDA :
            self.dec_channels *= 2
            
            self.EDA = []
            for i in range(self.model_length):
                module = EncoderDecoderAttention(self.enc_channels[self.model_length-i])
                self.add_module("EDA{}".format(i),module)
                self.EDA.append(module)

        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i],  padding_mode=padding_mode,dropout=dropout,activation=activation)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(
                self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], 
                kernel_size=self.dec_kernel_sizes[i],
                stride=self.dec_strides[i], 
                padding=self.dec_paddings[i], 
                output_padding=self.dec_output_paddings[i],
                activation=activation)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        if res_path == "None" : 
            self.use_respath=False

        if self.use_respath : 
            self.respaths = [] 
            for i in range(self.model_length) :
                module = ResPath(self.enc_channels[i])
                self.add_module("respath{}".format(i),module)
                self.respaths.append(module)

        ## Bottlenect
        self.bottleneck_type = bottleneck_type
        if bottleneck_type == 'None' :
            self.bottleneck = nn.Identity()
        elif bottleneck_type == 'GRU':
            # case for F = 513 
            self.bottleneck = nn.GRU(input_size  = 128*3,hidden_size = 64*3,num_layers = 2,bias=True,batch_first=True,bidirectional=True,dropout =dropout)
        elif bottleneck_type == 'LSTM':
            # case for F = 513 
            self.bottleneck = nn.LSTM(input_size  = 128*3,hidden_size = 64*3,num_layers = 2,bias=True,batch_first=True,bidirectional=True,dropout=dropout)
        elif bottleneck_type == 'TCN':
            self.bottleneck = TCN(c_in=128*3, c_out=[128*3,128*3])
        else :
            raise Exception("ERROR:UNET::bottleneck {} is not implemented".format(self.bottleneck_type))

        
        linear = nn.Conv2d(self.dec_channels[-1], 1, 1)
        self.mask_acti = None
        if mask_activation == 'Sigmoid' : 
            self.mask_acti = nn.Sigmoid()
        elif mask_activation == 'ReLU' : 
            self.mask_acti = nn.ReLU()
        elif mask_activation == 'Softplus':
            # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#torch.nn.Softplus
            self.mask_acti = nn.Softplus()
        elif mask_activation == "SiLU":
            self.mask_acti = nn.SiLU()
        elif mask_activation == 'none':
            self.mask_acti = nn.Identity()
        elif mask_activation == 'PReLU':
            self.mask_acti = nn.PReLU()
        else :
            raise Exception('ERROR:Unknown activation : ' + str(activation))

        self.add_module("linear", linear)
        self.padding_mode = padding_mode
        
        OA = OberservedAddition(method = OA_method,dim=OA_dim,factor=OA_factor)
        self.add_module('OA',OA)

    def forward(self, x):        
        # ipnut : [ Batch Channel Freq Time]

        # Time must be multiple of 16
        len_orig = x.shape[-1]
        need =  int(16*np.floor(len_orig/16)+16) - len_orig
        x = torch.nn.functional.pad(x,(0,need))
        # Encoder 
        x_skip = []
        for i, encoder in enumerate(self.encoders):
            if self.use_respath : 
                x_skip.append(self.respaths[i](x))
            else :
                x_skip.append(x)
            x = encoder(x)
            #print("x{}".format(i), x.shape)
        # x_skip : x0=input x1 ... x9

        #print("fully encoded ",x.shape)
        if self.bottleneck_type == 'GRU' or self.bottleneck_type == 'LSTM':
            # [B, C, F, T]
            B, C, F, T = x.shape
            x = torch.permute(x,(0,3,1,2))
            x = torch.reshape(x,(B,T,C*F))
            p,_ = self.bottleneck(x)
            p = torch.reshape(p,(B,T,C,F))
            p = torch.permute(p,(0,2,3,1))
        elif self.bottleneck_type == 'TCN':
            B, C, F, T = x.shape
            x = torch.reshape(x,(B,C*F,T))
            p = self.bottleneck(x)
            p = torch.reshape(p,(B,C,F,T))
        else :
            p = x
        
        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            #print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {x_skip[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")

            # last layer of Decorders
            if i == self.model_length - 1:
                break
            p = torch.cat([p, x_skip[self.model_length - 1 - i]], dim=1)

        #print('p : ' +str(p.shape))
        mask = self.linear(p)
        mask = self.mask_acti(mask)
        
        # Observation Addition
        mask_oberserved = torch.ones(mask.shape).to(self.device)
        mask = self.OA(mask,mask_oberserved)
        return mask[:,:,:,:len_orig]

    def set_size(self,depth=20):
        if depth == 10 : 
            self.enc_channels = [self.input_channels,
                                 self.model_complexity,
                                 self.model_complexity * 2,
                                 self.model_complexity * 2,
                                 self.model_complexity * 2,
                                 self.model_complexity * 2,
                                 ]
            self.enc_kernel_sizes = [(7, 5),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]
            self.enc_strides = [(2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 1)]
            self.enc_paddings = [(2, 1),
                                 None,
                                 None,
                                 None,
                                 None]

            self.dec_channels = [0,
                                 self.model_complexity * 2,
                                 self.model_complexity * 2,
                                 self.model_complexity * 2,
                                 self.model_complexity * 2,
                                 self.model_complexity * 2]

            self.dec_kernel_sizes = [(4, 3),
                                     (4, 4),
                                     (6, 4),
                                     (6, 4),
                                     (7, 5)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2)]

            self.dec_paddings = [(1, 1),
                                 (1, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1)]


        elif depth == 20 :
            self.enc_channels = [self.input_channels,
                                    self.model_complexity,
                                    self.model_complexity,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    128]

            self.enc_kernel_sizes = [   (7, 1),
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
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2,
                                    self.model_complexity * 2]

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

        else :
            raise Exception("ERROR::UNET::depth {} is not implemented".format(depth))
    
if __name__ == '__main__':
    path_data_sample = '/home/data/kbh/MCSE/CGMM_RLS_MPDR/train/SNR-5/noisy/011_011C0201.pt'
    input = torch.load(path_data_sample)
    input = input[:,:256,:]
    input = torch.sqrt(input[:,:,0]**2 + input[:,:,1]**2) 
    # batch
    input = torch.unsqueeze(input,dim=0)
    # channel
    input = torch.unsqueeze(input,dim=0)
    print(input.shape)

    model = Unet20()

    output = model(input)
    print(output.shape)