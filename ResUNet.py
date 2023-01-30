import torch
import torch.nn as nn
try : 
    from .UNet_m import *
except ImportError:
    from UNet_m import *
    
class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

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

class ResUNet(nn.Module) :
    def __init__(self, 
                 c_in = 4,
                 c_out = 1,
                 n_fft=512,
                 device="cuda:0",
                 output="mapping",
                 print_shape=False,
                 T=125
                 ):
        super().__init__()

        n_hfft = int(n_fft/2+1)

        self.print_shape=print_shape

        self.F = n_hfft
        self.T = T

        f_dim = 64

        ## Model Implementation


        # input layer
        self.layer_data = nn.Sequential(
            Encoder(c_in-2,20,(1,3),1,(0,1),1),
            nn.LayerNorm(n_hfft) 
        )
        self.layer_AF = nn.Sequential(
            Encoder(2,f_dim-20,(1,3),1,(0,1),1),
            nn.LayerNorm(n_hfft)
        )

        # Encoder
        encoders=[]
        encoders.append(ResBlock(f_dim))
        encoders.append(nn.Sequential(
                    Encoder(f_dim,f_dim,(3,3),
                    (2,1),(0,1),activation="PReLU"),
                    ResBlock(f_dim)))
        encoders.append(nn.Sequential(
                    Encoder(f_dim,f_dim,(3,3),
                    (2,1),(0,1),activation="PReLU"),
                    ResBlock(f_dim)))
        encoders.append(nn.Sequential(
                    Encoder(f_dim,f_dim,(3,3),
                    (2,1),(0,1),activation="PReLU"),
                    ResBlock(f_dim)))
        encoders.append(nn.Sequential(
                    Encoder(f_dim,f_dim,(3,3),
                    (2,1),(0,1),activation="PReLU"),
                    ResBlock(f_dim)))
        encoders.append(nn.Sequential(
                    Encoder(f_dim,f_dim,(3,3),
                    (2,1),(0,1),activation="PReLU"),
                    ResBlock(f_dim)))
        encoders.append(Encoder(f_dim,f_dim*2,(3,1),
                    (2,1),(0,0),activation="PReLU"))
        encoders.append(Encoder(f_dim*2,f_dim*6,(3,1),
                    (1,1),(0,0),activation="PReLU"))

        self.encoders=encoders
        for i,enc in enumerate(self.encoders) : 
            self.add_module("enc_{}".format(i),enc)

        # Decoder
        decoders=[]
        decoders.append(Decoder(f_dim*6,f_dim*2,(3,1),
            (1,1),(0,0),activation="PReLU"))
        decoders.append(Decoder(f_dim*2,f_dim,(4,1),
            (2,1),(1,0),output_padding=(1,0),activation="PReLU"))
        decoders.append(nn.Sequential(
            ResBlock(f_dim),
            Decoder(f_dim,f_dim,(4,1),
            (2,1),(1,0),output_padding=(1,0),activation="PReLU")))
        decoders.append(nn.Sequential(
            ResBlock(f_dim),
            Decoder(f_dim,f_dim,(4,1),
            (2,1),(1,0),output_padding=(1,0), activation="PReLU")))
        decoders.append(nn.Sequential(
            ResBlock(f_dim),
            Decoder(f_dim,f_dim,(4,1),
            (2,1),(1,0),output_padding=(1,0),activation="PReLU")))
        decoders.append(nn.Sequential(
            ResBlock(f_dim),
            Decoder(f_dim,f_dim,(4,1),
            (2,1),(0,0),output_padding=(0,0),activation="PReLU")))
        decoders.append(nn.Sequential(
            ResBlock(f_dim),
            Decoder(f_dim,f_dim,(4,1),
            (2,1),(1,0),output_padding=(1,0),activation="PReLU")))

        decoders.append(ResBlock(f_dim))

        self.decoders=decoders
        for i,dec in enumerate(self.decoders) : 
            self.add_module("dec_{}".format(i),dec)

        self.len_model = len(encoders)

        # Residual Path
        res_paths = []
        res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(f_dim*2,f_dim*2,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(f_dim*6,f_dim*6,1,1,0,1,activation="PReLU"))

        self.res_paths = res_paths
        for i,res_path in enumerate(self.res_paths) : 
            self.add_module("res_path_{}".format(i),res_path)

        # Bottlenect
        self.bottleneck = nn.LSTM(f_dim*6,f_dim*10,3,batch_first=True,proj_size=f_dim*6)

        # output layer
        self.out_layer = nn.ConvTranspose2d(f_dim,2,(3,1),stride=1,padding=(1,0),dilation=1,output_padding=(0,0))

        self.last_activation=nn.Tanh()

    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        self.apply(init_weights)

    def forward(self,input):
        ## ipnut : [ Batch Channel Freq Time]
        # reshape
        # [ B C T F]
        feature = torch.permute(input[:,:-2,:,:],(0,1,3,2))
        AF = torch.permute(input[:,-2:,:,:],(0,1,3,2))

        # input
        feature = self.layer_data(feature)
        AF = self.layer_AF(AF)

        # reshape
        feature = torch.permute(feature,(0,1,3,2))
        AF = torch.permute(AF,(0,1,3,2))
        x = torch.cat((feature,AF),dim=1)

        ## Encoder
        res=[]
        for i,enc in enumerate(self.encoders):
            x = enc(x)
            if self.print_shape :
                print("x_{} : {}".format(i,x.shape))
            res.append(x)

        ## bottleneck
        # [B,C,1,T] -> [B,C,T]
        x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[3]))
        # [B,C,T] -> [B,T,C]
        x = torch.permute(x,(0,2,1))
        #print("bottle in : {}".format(x.shape))

        x = self.bottleneck(x)[0]

        # [B,T,C] -> [B,C,T]
        x = torch.permute(x,(0,2,1))
        # [B,C,T] -> [B,C,1,T]
        x = torch.reshape(x,(x.shape[0],x.shape[1],1,x.shape[2]))

        ## ResPath
        for i,res_path in enumerate(self.res_paths) : 
            res[i] = res_path(res[i])

        ## Decoder
        y = x

        for i,dec in enumerate(self.decoders) : 
            if self.print_shape :
                print("y {} += r_{} : {}".format(y.shape,i,res[-1-i].shape))
            y  = torch.add(y,res[-1-i])
            y = dec(y)
            if self.print_shape :
                print("y_{} : {}".format(i,y.shape))

        ## output
        y = self.out_layer(y)

        mask = self.last_activation(y)+1e-13

        output = torch.mul(mask,input[:,:2,:,:])
        return output


"""
    Encoder-Decoder works only on Freq
    Time axis for RNN
"""
class ResUNetOnFreq(nn.Module) :
    def __init__(self, 
                 c_in = 4,
                 c_out = 1,
                 n_fft=512,
                 device="cuda:0",
                 print_shape=False,
                 T=125,
                 n_block = 5,
                 activation = "Softplus" , 
                 Softplus_thr = 20,
                 norm = "BatchNorm2d",
                 dropout = 0.0
                 ):
        super().__init__()

        n_hfft = int(n_fft/2+1)

        self.print_shape=print_shape

        self.F = n_hfft
        self.T = T

        #f_dim = 30
        f_dim = 30

        if n_block < 2 :
            raise Exception("ERROR::ResUnetOnFreq : n_block({}) < 2".fomrat(n_block))

        ## Model Implementation

        # input layer
        self.layer_input = nn.Sequential(
            Encoder(c_in,f_dim,(1,3),1,(0,1),1),
            nn.LayerNorm(n_hfft) 
        )

        # Encoder
        encoders=[]
        encoders.append(ResBlock(f_dim))
        for i in range(n_block) :
            encoders.append(nn.Sequential(
                    Encoder(f_dim,f_dim,(3,1),
                    (2,1),(0,0),activation="PReLU",norm=norm),
                    ResBlock(30)))
        encoders.append(Encoder(f_dim,64,(3,1),
                    (2,1),(0,0),activation="PReLU",norm=norm))
        encoders.append(Encoder(64,192,(3,1),
                    (1,1),(0,0),activation="PReLU",norm=norm))

        self.encoders=encoders
        for i,enc in enumerate(self.encoders) : 
            self.add_module("enc_{}".format(i),enc)

        # Decoder
        decoders=[]
        decoders.append(Decoder(192,64,(3,1),
            (1,1),(0,0),activation="PReLU",norm=norm))
        decoders.append(Decoder(64,f_dim,(4,1),
            (2,1),(1,0),output_padding=(1,0),activation="PReLU",norm=norm))
        
        for i in range(n_block-2) :
            decoders.append(nn.Sequential(
                ResBlock(f_dim),
                Decoder(f_dim,f_dim,(4,1),
                (2,1),(1,0),output_padding=(1,0),activation="PReLU",norm=norm)))
        decoders.append(nn.Sequential(
                ResBlock(f_dim),
                Decoder(f_dim,f_dim,(5,1),
                (2,1),(1,0),output_padding=(1,0),activation="PReLU",norm=norm)))
        decoders.append(nn.Sequential(
                ResBlock(f_dim),
                Decoder(f_dim,f_dim,(4,1),
                (2,1),(1,0),output_padding=(1,0),activation="PReLU",norm=norm)))
        decoders.append(ResBlock(f_dim))

        self.decoders=decoders
        for i,dec in enumerate(self.decoders) : 
            self.add_module("dec_{}".format(i),dec)

        self.len_model = len(encoders)

        # Residual Path
        res_paths = []
        res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation="PReLU"))
        for i in range(n_block) : 
            res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(64,64,1,1,0,1,activation="PReLU"))
        res_paths.append(Encoder(192,192,1,1,0,1,activation="PReLU"))

        self.res_paths = res_paths
        for i,res_path in enumerate(self.res_paths) : 
            self.add_module("res_path_{}".format(i),res_path)

        # Bottlenect
        self.bottleneck = nn.LSTM(192,300,3,batch_first=True,proj_size=192,dropout=dropout)

        # output layer
        self.out_layer = nn.ConvTranspose2d(f_dim,c_out,(3,1),stride=1,padding=(1,0),dilation=1,output_padding=(0,0))

        if activation == "Softplus" : 
            self.activation_mask = nn.Softplus(threshold=Softplus_thr)
        elif activation == "Sigmoid" : 
            self.activation_mask = nn.Sigmoid()
        else : 
            self.activation_mask = nn.Softplus()

    def forward(self,input):
        ## ipnut : [ Batch Channel Freq Time]
        # reshape
        # [ B C T F]
        feature = torch.permute(input[:,:,:,:],(0,1,3,2))
        feature = self.layer_input(feature)

        # reshape
        x = torch.permute(feature,(0,1,3,2))

        ## Encoder
        res=[]
        for i,enc in enumerate(self.encoders):
            x = enc(x)
            if self.print_shape : 
                print("x_{} : {}".format(i,x.shape))
            res.append(x)


        ## bottleneck
        # [B,C,1,T] -> [B,C,T]
        x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[3]))
        # [B,C,T] -> [B,T,C]
        x = torch.permute(x,(0,2,1))
        #print("bottle in : {}".format(x.shape))

        x = self.bottleneck(x)[0]

        # [B,T,C] -> [B,C,T]
        x = torch.permute(x,(0,2,1))
        # [B,C,T] -> [B,C,1,T]
        x = torch.reshape(x,(x.shape[0],x.shape[1],1,x.shape[2]))

        ## ResPath
        for i,res_path in enumerate(self.res_paths) : 
            res[i] = res_path(res[i])

        ## Decoder

        y = x

        for i,dec in enumerate(self.decoders) : 
            if self.print_shape : 
                print("y {} += r_{} : {}".format(y.shape,i,res[-1-i].shape))
            y  = torch.add(y,res[-1-i])
            y = dec(y)
            if self.print_shape : 
                print("y_{} : {}".format(i,y.shape))

        ## output
        
        output = self.out_layer(y)
        return self.activation_mask(output)
    
    def output(self,mask,feature):
        return mask * feature[:,:2]

"""
pre-model for ver3
"""
class ResUNetOnFreq2(nn.Module) :
    def __init__(self, 
                 c_in = 1,
                 c_out = 1,
                 n_fft=512,
                 device="cuda:0",
                 print_shape=False,
                 n_block = 5,
                 activation = "Softplus" , 
                 bottleneck = "LSTM",
                 Softplus_thr = 20,
                 norm = "BatchNorm2d",
                 dropout = 0.0,
                 activation_layer = "PReLU",
                 multi_scale = False
                 ):
        super().__init__()

        n_hfft = int(n_fft/2+1)

        self.print_shape=print_shape
        self.activation = activation
        self.multi_scale = multi_scale
        upscale = [1,2.2, 4.5, 9, 18.4, 36.8]

        self.F = n_hfft
        f_dim = 30

        if n_block < 2 :
            raise Exception("ERROR::ResUnetOnFreq : n_block({}) < 2".fomrat(n_block))

        ## Model Implementation

        # input layer
        self.layer_input = nn.Sequential(
            Encoder(c_in,f_dim,(1,3),1,(0,1),1),
            nn.LayerNorm(n_hfft) 
        )

        # Encoder
        encoders=[]
        encoders.append(ResBlock(f_dim))
        for i in range(n_block) :
            encoders.append(nn.Sequential(
                    Encoder(f_dim,f_dim,(3,1),
                    (2,1),(0,0),activation=activation_layer,norm=norm),
                    ResBlock(30)))

        self.encoders=encoders
        for i,enc in enumerate(self.encoders) : 
            self.add_module("enc_{}".format(i),enc)

        # Multi-Scale Feature
        if multi_scale : 
            ms = []
            for i in range(n_block):
                ms.append(nn.Sequential(
                    #nn.Conv2d(30,30,(2**(n_block+1-i),1),stride=(2**(n_block-i),1),padding=(1,0))
                    nn.AvgPool2d((2**(n_block+1-i),1),stride=(2**(n_block-i),1),padding=(1,0))
                ))
            self.ms = ms
            for i,i_ms in enumerate(self.ms) : 
                self.add_module("ms_{}".format(i),i_ms)
                
            self.ms_block = MultiScaleConvBlock(f_dim)

            self.upsample = []
            for scale in upscale : 
                self.upsample.append(nn.Sequential(
                                nn.Upsample(scale_factor=(scale,1), mode='nearest'),
                                nn.Sigmoid()
                ))
            for i,i_up in enumerate(self.upsample) : 
                self.add_module("up_{}".format(i),i_up)

        # Decoder
        decoders=[]
        for i in range(n_block-2) :
            decoders.append(nn.Sequential(
                ResBlock(f_dim),
                Decoder(f_dim,f_dim,(4,1),
                (2,1),(1,0),output_padding=(1,0),activation=activation_layer,norm=norm)))
        decoders.append(nn.Sequential(
                ResBlock(f_dim),
                Decoder(f_dim,f_dim,(5,1),
                (2,1),(1,0),output_padding=(1,0),activation=activation_layer,norm=norm)))
        decoders.append(nn.Sequential(
                ResBlock(f_dim),
                Decoder(f_dim,f_dim,(4,1),
                (2,1),(1,0),output_padding=(1,0),activation=activation_layer,norm=norm)))
        decoders.append(ResBlock(f_dim))

        self.decoders=decoders
        for i,dec in enumerate(self.decoders) : 
            self.add_module("dec_{}".format(i),dec)

        self.len_model = len(encoders)

        # Residual Path
        res_paths = []
        res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation=activation_layer))
        for i in range(n_block) : 
            res_paths.append(Encoder(f_dim,f_dim,1,1,0,1,activation=activation_layer))

        self.res_paths = res_paths
        for i,res_path in enumerate(self.res_paths) : 
            self.add_module("res_path_{}".format(i),res_path)

        # Bottleneck
        if bottleneck == "LSTM" : 
            self.bottleneck = LSTMBlock(210,300,n_layer=3,dropout=dropout)
        elif bottleneck == "FTGRU" : 
            bottleneck_hidden = 256
            bottleneck_channel = f_dim*2
            self.bottleneck = nn.Sequential(
                FGRUBlock(f_dim, bottleneck_hidden, bottleneck_channel),
                TGRUBlock(bottleneck_channel, bottleneck_hidden, f_dim)
            )            
        else :
            self.bottleneck = LSTMBlock(210,300,n_layer=3,dropout=dropout)
        # output layer
        self.out_layer = nn.ConvTranspose2d(f_dim,c_out,(3,1),stride=1,padding=(1,0),dilation=1,output_padding=(0,0))

        if activation == "Softplus" : 
            self.activation_mask = nn.Softplus(threshold=Softplus_thr)
        elif activation == "Sigmoid" : 
            self.activation_mask = nn.Sigmoid()
        elif activation == "Tanh" : 
            self.activation_mask = nn.Tanh()
        elif activation == "Identity" : 
            self.activation_mask = nn.Identity()
        elif activation == "MEA" : 
            if c_in == 1 :
                raise Exception("ERROR::ResUnetOnFreq::feature must be complex")
            self.activation_mask = MEA(in_channels = c_out)
            self.add_module("MEA",self.activation_mask)
        else : 
            self.activation_mask = nn.Softplus()

    def forward(self,input):
        ## ipnut : [ Batch Channel Freq Time]
        # reshape
        # [ B C T F]
        feature = torch.permute(input[:,:,:,:],(0,1,3,2))
        feature = self.layer_input(feature)

        # reshape
        x = torch.permute(feature,(0,1,3,2))

        ## Encoder
        res=[]
        ms = None
        for i,enc in enumerate(self.encoders):
            x = enc(x)
            if self.print_shape : 
                print("x_{} : {}".format(i,x.shape))
            res.append(x)

            # multi-scale
            if self.multi_scale and i < len(self.ms) :
                if ms is None : 
                    ms = self.ms[i](x)
                else : 
                    ms += self.ms[i](x)
                if self.print_shape : 
                    print("ms_{} : {}".format(i,ms.shape))
                ms = self.ms_block(ms)



        ## bottleneck
        x = self.bottleneck(x)[0]

        ## ResPath
        for i,res_path in enumerate(self.res_paths) : 
            res[i] = res_path(res[i])

        ## Decoder
        y = x

        for i,dec in enumerate(self.decoders) : 
            if self.print_shape : 
                print("y {} += r_{} : {}".format(y.shape,i,res[-1-i].shape))
            y  = torch.add(y,res[-1-i])
            y = dec(y)
            if self.print_shape : 
                print("y_{} : {}".format(i,y.shape))

        ## output
        output = self.out_layer(y)
        return self.activation_mask(output)

    def output(self,mask,feature):

        if self.activation == "MEA" : 
            return self.last_activation.output(mask,feature)
        else :
            return mask * feature[:,:2]
