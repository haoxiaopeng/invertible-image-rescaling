from LBSign import LBSign
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from FrEIA.framework import *
from FrEIA.modules import *
#from cbn_layer import *
#from subnet_coupling import *
#import data
#import config as c
from torch.autograd import Variable


class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
        bias=True, weight_norm=True, scale=False):
        """Intializes a Conv2d augmented with weight normalization.
        (See torch.nn.utils.weight_norm for detail.)
        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            kernel_size: size of convolving kernel.
            stride: stride of convolution.
            padding: zero-padding added to both sides of input.
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()

        if weight_norm:
            self.conv = nn.utils.weight_norm(
                nn.Conv2d(in_dim, out_dim, kernel_size,
                    stride=stride, padding=padding, bias=bias))
            if not scale:
                self.conv.weight_g.data = torch.ones_like(self.conv.weight_g.data)
                self.conv.weight_g.requires_grad = False    # freeze scaling
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size,
                stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.conv(x)


class HaarConv(nn.Module):

    def __init__(self):
        super(HaarConv, self).__init__()

        self.in_channels = 3
        self.fac_fwd = 0.25
        self.haar_weights = torch.ones(4,1,2,2)
        self.haar_weights[1, :, 0, 1] = -1
        self.haar_weights[1, :, 1, 1] = -1

        self.haar_weights[2, :, 1, 0] = -1
        self.haar_weights[2, :, 1, 1] = -1

        self.haar_weights[3, :, 1, 0] = -1
        self.haar_weights[3, :, 0, 1] = -1
        #self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        #self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x):
        out = F.conv2d(x, self.haar_weights, bias=None, stride=2)
        out = out * self.fac_fwd
        return out

class HaarConv_inv(nn.Module):

    def __init__(self):
        super(HaarConv_inv, self).__init__()

        self.in_channels = 3
        self.fac_fwd = 0.25
        self.haar_weights = torch.ones(4,1,2,2)
        self.haar_weights[1, :, 0, 1] = -1
        self.haar_weights[1, :, 1, 1] = -1

        self.haar_weights[2, :, 1, 0] = -1
        self.haar_weights[2, :, 1, 1] = -1

        self.haar_weights[3, :, 1, 0] = -1
        self.haar_weights[3, :, 0, 1] = -1
        #self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)

        self.haar_weights.requires_grad = False

    def forward(self, x):
        out = F.conv_transpose2d(x, self.haar_weights, bias=None, stride=2)

        return out



class ChannelwiseAdditiveCoupling(nn.Module):
    def __init__(self, in_dim, out_dim):
        """Initializes a ChannelwiseAdditiveCoupling.
        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            hps: the set of hyperparameters.
        """
        super(ChannelwiseAdditiveCoupling, self).__init__()

        self.in_bn = nn.BatchNorm2d(in_dim, affine=False, track_running_stats=True)
        self.block = nn.Sequential(
            #nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, (3,3),
                    stride=1, padding=1, bias=True))
            #WeightNormConv2d(in_dim, out_dim, (3, 3), stride=1, padding=1,
            #                 bias=True,  scale=True))
        self.out_bn = nn.BatchNorm2d(out_dim, affine=False)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """

        #x = self.in_bn(x)
        #print("self.block",x)

        x = self.block(x)

        #x = self.out_bn(x)

        return x



class InvBlock_f(nn.Module):
    def __init__(self, in_dim, out_dim):
        """Initializes a RealNVP.
        Args:
            datainfo: information of dataset to be modeled.
            prior: prior distribution over latent space Z.
            hps: the set of hyperparameters.
        """
        super(InvBlock_f, self).__init__()

        self.scale_factor = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale_factor.data.fill_(1.0)        
        self.bais_factor = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bais_factor.data.fill_(0.0)        

        self.LH = nn.Conv2d(in_dim, out_dim, (3,3),stride=1, padding=1, bias=True)
        self.HL = nn.Conv2d(out_dim, in_dim, (3,3),stride=1, padding=1, bias=True)
        self.m = nn.Sigmoid()
        self.p_conv = nn.Conv2d(in_dim, out_dim, (3, 3), stride=1, padding=1,bias=True)
    def forward(self, h1, h2):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """

        h1_1 = self.HL(h2)
        y1 = h1_1 + h1
        h_m1 = self.p_conv(y1)
        h_m1 = torch.exp(h_m1)
        h_m1 = self.m(h_m1)
        h_m2 = torch.mul(h2,  (h_m1 * self.scale_factor + self.bais_factor))
        #print(h_m2[0][0])
        #self.scale_factor.register_hook(lambda x: print('sca',x))
        #self.h_m2.register_hook(lambda x: print('m2',x.shape))
        #self.h_m1.register_hook(lambda x: print('m1',x.shape))
        y2 = self.LH(y1) + h_m2

        return y1, y2

class InvBlock_g(nn.Module):
    def __init__(self, in_dim, out_dim):
        """Initializes a RealNVP.
        Args:
            datainfo: information of dataset to be modeled.
            prior: prior distribution over latent space Z.
            hps: the set of hyperparameters.
        """
        super(InvBlock_g, self).__init__()
        self.scale_factor = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale_factor.data.fill_(1.0)        
        self.bais_factor = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bais_factor.data.fill_(0.0)        
        self.LH = nn.Conv2d(in_dim, out_dim, (3,3),stride=1, padding=1, bias=True)
        self.HL = nn.Conv2d(out_dim, in_dim, (3,3),stride=1, padding=1, bias=True)
        self.m = nn.Sigmoid()
        self.p_conv = nn.Conv2d(in_dim, out_dim, (3, 3), stride=1, padding=1,bias=True)
        
    def forward(self, y1, y2):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """

        h2_1 = self.LH(y1)
        y2_2 = y2 - h2_1
        h_m1 = self.p_conv(y1)
        h_m1 = torch.exp(h_m1)
        h_m1 = self.m(h_m1)
        n,c,h,w = h_m1.shape
        for l in range(n):
            for k in range(c):
                if torch.sum(h_m1[l][k])==0:
                    print("h_m1 is nan")
        #print(self.scale_factor,self.bais_factor)

        #print('h_m1',h_m1[0][0])
        h_m2 = h_m1 * self.scale_factor + self.bais_factor
        #print('h_m2',h_m2[0][0])
        h2 = torch.div(y2_2 , h_m2)
        #print('h2',h2[0][0])


        h1 = y1 - self.HL(h2)

        return h1, h2

class IIR(nn.Module):
    def __init__(self,  in_dim, out_dim, prior):
        """Initializes a RealNVP.
        Args:
            datainfo: information of dataset to be modeled.
            prior: prior distribution over latent space Z.
            hps: the set of hyperparameters.
        """
        super(IIR, self).__init__()
        # self.datainfo = datainfo
        self.prior = prior
        # self.hps = hps
        #
        # chan = datainfo.channel
        # size = datainfo.size
        # dim = hps.base_dim
        #
        self.find_nearest = LBSign.apply

        inv_blocks = 8
        self.haar = HaarConv()
        self.haar_inv = HaarConv_inv()


        self.inv_block_f = nn.ModuleList(
            [InvBlock_f(in_dim, out_dim)
             for _ in range(inv_blocks)])

        self.inv_block_g = nn.ModuleList(
            [InvBlock_g(in_dim, out_dim)
             for _ in range(inv_blocks)])
        '''
        self.Loos_D = nn.ModuleList(
            nn.Conv2d(in_dim, 64, (3,3),
                    stride=1, padding=1, bias=True),
            nn.Conv2d(64, 64, (3,3),
                    stride=1, padding=1, bias=True),
            nn.Conv2d(64, 128, (3,3),
                    stride=1, padding=1, bias=True),
            nn.Conv2d(128, 128, (3, 3),
                    stride=1, padding=1, bias=True),
            nn.Conv2d(128, 256, (3, 3),
                    stride=1, padding=1, bias=True),
            nn.Conv2d(256, 256, (3, 3),
                    stride=1, padding=1, bias=True),
            nn.Conv2d(256, 512, (3, 3),
                    stride=1, padding=1, bias=True),
            nn.Conv2d(512, 512, (3, 3),
                    stride=1, padding=1, bias=True),
            nn.Linear(512, 100),
            nn.Linear(100, 100))
        '''
        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10000, 20000, 30000, 40000], gamma=0.5)

        # self.scale_factor = Variable(torch.Tensor[2], requires_grad=True)

    # def D_basic(self, netD, real, fake):

    def find_neares1t(self,input_img):
        output = torch.clamp((input_img*255),0.,255.).round()
        output = output/255.
        return output

    def sample(self,size):
        """Generates samples.
        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """

        #C = 9
        #H = W = 72
        #size = 1
        z = self.prior.sample(size)
        return z
    def hook(self,grad):
        self.grad_for_encoder = grad
        return grad
    def forward(self, x):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """

        imgr_y = self.haar(x)
        h1,h2 = torch.split(imgr_y,[1,3],dim=1)
        n,c,h,w = h2.shape
        #h2.register_hook(lambda x: print('haar',x))
        #for l in range(n):
        #    if torch.sum(h2[l][0]) ==0 or torch.sum(h2[l][1])==0 or torch.sum(h2[l][2])==0:
        #        print('haar is nan')
        #        return 0,h1,x
        for i in range(len(self.inv_block_f)):
            h1, h2 = self.inv_block_f[i](h1, h2)
            h1.register_hook(lambda g: print('f',g))
           # for l in range(n):
           #     if torch.sum(h2[l][0]) ==0 or torch.sum(h2[l][1])==0 or torch.sum(h2[l][2])==0:
           #         print('f is nan',i)
           #         return 0,h1,x
        gg1=h1
        g1 = self.find_nearest(gg1)
        g1.register_hook(lambda g: print('ste',g))
        g2=self.sample(h2.shape)
        for i in range(len(self.inv_block_g)):
            g1, g2 = self.inv_block_g[i](g1, g2)
            g1.register_hook(lambda g: print('g',g))
            #for l in range(n):
            #    if torch.sum(g2[l][0]) ==0 or torch.sum(g2[l][1])==0 or torch.sum(g2[l][2])==0:
            #        print('g is nan',i)
            #        return 0,g1,x

        imgy_s = torch.cat((g1,g2),1)
        imgy_s = self.haar_inv(imgy_s)
        return 1,h1, imgy_s




