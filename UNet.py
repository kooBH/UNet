import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class UNet(nn.Module):
    def __init__(self, 
                 encoders,
                 decoders,
                 bottlenecks=None,
                 res_paths = None,
                 device="cuda:0"
                 ):
        super().__init__()

        self.device = device

        ## Exception Check
        if len(encoders) != len(decoders) :
            raise Exception("ERROR::unmatched legnth : enc {} != dec {}".format(len(encoders),len(decoders)))
        else :
            self.len_model = len(encoders)

        self.print_shape = print_shape

        self.encoders = encoders
        self.decoders = decoders
        for i in range(len(encoders)) : 
            module = self.encoders[i]
            self.add_module("encoder_{}".format(i),module)
        for i in range(len(decoders)) : 
            module = self.decoders[i]
            self.add_module("decoder_{}".format(i),module)

        # Residual Path
        self.res_paths = []
        if res_paths is not None  :
            if (len(res_paths) != self.len_model -1) :
                raise Exception("ERROR::unmatched res_path : {} != {}".format(len(res_paths),self.len_model-1))
            else :
                for i in range(len(res_paths)):
                    module = res_paths[i]
                    self.add_module("res_path{}".format(i),module)
                    self.res_paths.append(module)
        # default : skip connection
        else :
            for i in range(self.len_model-1):
                module = nn.Identity()
                self.add_module("res_path{}".format(i),module)
                self.res_paths.append(module)
        # Dummy
        module = nn.Identity()
        self.add_module("res_path{}".format(i+1),module)
        self.res_paths.append(module)
            
        ## Bottlenect
        self.bottlenecks = []
        if bottlenecks is not None :
            for i in range(len(bottlenecks)):
                module = bottlenecks[i]
                self.add_module("bottleneck{}".format(i),module)
                self.bottlenecks.append(module)
        else :
            module = nn.Identity()
            self.add_module("bottleneck{}".format(0),module)
            self.bottlenecks.append(module)

        bottleneck_channel = encoders[-1].conv.out_channels
        
        linear = nn.Conv2d(1, 1, 1)
        self.add_module("linear", linear)
        self.activation_mask = nn.Sigmoid()
        
    def forward(self, x):        
        # ipnut : [ Batch Channel Freq Time]

        # Time must be multiple of 16
        """
        len_orig = x.shape[-1]
        need =  int(16*np.floor(len_orig/16)+16) - len_orig
        x = torch.nn.functional.pad(x,(0,need))
        """

        # Encoder 
        x_skip = []
        for i, encoder in enumerate(self.encoders):
            x_skip.append(self.res_paths[i](x))
            x = encoder(x)
            if self.print_shape : 
                print("Encoder {} : {}".format(i,x.shape))

        p = x
        for i, bottleneck in enumerate(self.bottlenecks):
            p  =  bottleneck(p)
            if self.print_shape : 
                print("bottleneck {} : {}".format(i,p.shape))
        
        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if self.print_shape : 
                print("Decoder {} : {}".format(i,p.shape))
            # last layer of Decorders
            if i == self.len_model- 1:
                break
            p = torch.cat([p, x_skip[self.len_model - 1 - i]], dim=1)
            if self.print_shape : 
                print("Decoder cat {} : {}".format(i,p.shape))

        mask = self.linear(p)
        mask = self.activation_mask(mask)
        
        return mask

