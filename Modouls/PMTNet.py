

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.checkpoint as cp
import collections
import sys
import math
import numpy as np

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

def get_Mirror_same_padding_conv2d(image_size=None):
    return Conv2dMirrorSamePadding

class Conv2dMirrorSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw) # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],mode='reflect')
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
###### Layer 
def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)


def conv_downsample(in_channels, out_channels):
    return nn.Conv2d(in_channels,out_channels,kernel_size=2,
                     stride=2, padding=0, bias=False)
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,dilation=1,mid=None):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        if mid is None:
            mid_channels = out_channels // 2
        else:
            mid_channels = mid
        m['conv1'] = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(mid_channels, mid_channels, kernel_size=7, stride=1, padding=3*dilation, bias=False,dilation=dilation)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x) 
        return out


    
    
class Convunit(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,dilation=1,mid=None,has_se=True,skip=True,drop=0.2):
        super(Convunit,self).__init__()
        self.has_se=has_se
        self.id_skip=skip
        self.drop_rate=drop
        m = OrderedDict()
        if mid is None:
            mid_channels = out_channels // 2
        else:
            mid_channels = mid
          
        m['conv1'] = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['pad2'] = nn.ReflectionPad2d((kernel_size-1)//2 * dilation)
        m['conv2'] = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, 
                               stride=1, padding=0, bias=False,dilation=dilation)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        if self.has_se:
            self.se = SELayer(out_channels)
    def forward(self, x):
        out = self.group1(x) 
        if self.has_se:
            out = self.se(out)
        if self.id_skip :
            # The combination of skip connection and drop connect brings about stochastic depth.
            if self.drop_rate:
                out = drop_connect(out, p=self.drop_rate, training=self.training)
            out = x + out  # skip connection
        return out
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LFTM(nn.Module):
    def __init__(self, channel, reduction=4):
        super(LFTM, self).__init__()
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.proj = nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, bias=False)
        self.xconv = nn.Conv2d(channel// reduction, channel // reduction, kernel_size=1, stride=1, bias=False)
        self.expendc = nn.Conv2d(channel// reduction, 4 , kernel_size=1, stride=1, bias=False)
        self._relu = nn.ReLU(inplace=True)
        self.upt = nn.Upsample(scale_factor=2, mode='nearest')
        self.selayer = SELayer(channel+4)
        self.oconv = nn.Conv2d(channel+4, channel , kernel_size=1, stride=1, bias=False)
        
        
        
    def forward(self, x):
        out = self.avg_pool(x)
        out = self._relu( self.proj(out) )
        out = self._relu(self.xconv(out))
        out = self.expendc(out)
        out = self.selayer( torch.cat([ x, self.upt(out)  ],1) ) 
        return self.oconv(out)
    
class PTFBlock(nn.Module):
    def __init__(self, channel, k=5):
        super(PTFBlock, self).__init__()
        self.upsample = nn.Upsample( scale_factor=2, mode='bilinear',align_corners=True)
        mid = channel * 2
        self.expand = nn.Conv2d(3, channel, kernel_size=1, stride=1, bias=False)
        Conv2d = get_Mirror_same_padding_conv2d(image_size=None)
        self.dwconv1 = Conv2d(in_channels=mid , out_channels=mid,groups=mid, kernel_size=k, bias=False)
        self.dwconv2 = Conv2d(in_channels=mid , out_channels=mid,groups=mid, kernel_size=k, bias=False)
        self.proj1 = nn.Conv2d(mid, channel, kernel_size=1, stride=1, bias=False)
        self.proj2 = nn.Sequential(conv3x3(channel *2 ,channel-3),nn.Tanh())
        self.selayer1 = SELayer(mid)
        self.selayer2 = SELayer(channel)
        self._swish = MemoryEfficientSwish()#Swish()
        
        
    def forward(self, fea,xu,yd):
        cy = xu - self.upsample(yd)
        fea = self.upsample(fea)
        tex = self.expand(cy) 
        tex = torch.cat([tex,fea],1)
        tex = self._swish(self.dwconv2( self._swish(self.dwconv1(tex))  ))
        tex = self.selayer1(tex)
        tex = fea+self.proj1(tex) 
        tex = torch.cat([tex , self.selayer2(fea)],1)
        tex = self.proj2( tex )
        tex = torch.cat([xu,tex],1)
        return tex
    
class RDTBlock(nn.Module):
    def __init__(self, channels=48,d1 = 1,d2 = 1 ,k=5):
        super(RDTBlock, self).__init__()
        mid = channels * 2
        Conv2d = get_Mirror_same_padding_conv2d(image_size=None)
        self.convtotwo = Conv2d(in_channels=channels, out_channels=mid, kernel_size=1, bias=False)
        self.convtotwo2 = Conv2d(in_channels=channels, out_channels=mid, kernel_size=1, bias=False)
        self.depthwise_conv1 = Conv2d(
            in_channels=mid, out_channels=mid, groups=mid,  # groups makes it depthwise
            kernel_size=k, stride=1,dilation=d1, bias=False)
        self.project_conv1 = Conv2d(in_channels=mid, out_channels=channels, kernel_size=1, bias=False)
        self.selayer1 = SELayer(mid)
        self.depthwise_conv2 = Conv2d(
            in_channels=mid, out_channels=mid, groups=mid,  # groups makes it depthwise
            kernel_size=k, stride=1,dilation =d2, bias=False)
        self.selayer2 = SELayer(mid)
        self.project_conv2 = Conv2d(in_channels=mid, out_channels=channels, kernel_size=1, bias=False)
        self._swish = MemoryEfficientSwish()  #Swish()
        self.lftm1 = LFTM(channels)
        self.upsample = nn.Upsample( scale_factor=2, mode='bilinear')
            
    def forward(self, x):
        out = self._swish(self.convtotwo(x))
        out = self._swish( self.depthwise_conv1(out) )
        out = self.selayer1(out)
        out = self.project_conv1(out) 
        out = x + out 
        out2 = self._swish(self.convtotwo2(out))
        out2 = self._swish(self.depthwise_conv2(out2) )
        out2 = self.selayer2(out2)
        out2 = self.project_conv2(out2) 
        out2 = out + self.lftm1(out2)
        return out2

    
class confidentBlock(nn.Module):
    def __init__(self, channels=32):
        super(confidentBlock, self).__init__()
        self.Cfblock1 = SELayer(channels,4)

        self.upsample = nn.Upsample( scale_factor=2, mode='bilinear')
            
    def forward(self, x):
        out = self.upsample(  self.Cfblock1(x)  )
        return out

class PMTNet(nn.Module):
    def __init__(self,channels=48,istrain=True):
        super(PMTNet, self).__init__()

        self.istrain = istrain
        self.relu = nn.ReLU(True)

        self.conv_in3 = nn.Sequential(conv3x3(3,channels))
        

            
        
        
        self.iblock1 = RDTBlock(channels)
        self.iblock2 = RDTBlock(channels,2,3)
        self.iblock3 = RDTBlock(channels,5,7)
        self.iblock4 = RDTBlock(channels)
        
        
        self.iblock1_2 = RDTBlock(channels)
        self.iblock2_2 = RDTBlock(channels,1,2)
        self.iblock3_2 = RDTBlock(channels,3,5)
        self.iblock4_2 = RDTBlock(channels,7,1)
        self.iblock5_2 = RDTBlock(channels)
        
        
        self.iblock1_3 = RDTBlock(channels)
        self.iblock2_3 = RDTBlock(channels)
        self.iblock3_3 = RDTBlock(channels,2,3)
        self.iblock4_3 = RDTBlock(channels,5,7)
        self.iblock5_3 = RDTBlock(channels,9)
        self.iblock6_3 = RDTBlock(channels)
        
        self.fusion2 = PTFBlock(channels)
        self.fusion1 = PTFBlock(channels)

        
        self.midout =nn.Sequential(conv3x3(channels,channels-3),nn.Tanh())
        self.midout3 =nn.Sequential(conv3x3(channels,channels-3),nn.Tanh())
        self.conv_out = nn.Sequential(conv3x3(channels,3),nn.Tanh()) #,nn.ReLU(True)
        self.conv_out2 = nn.Sequential(conv3x3(channels,3),nn.Tanh())
        self.conv_out3 = nn.Sequential(conv3x3(channels,3),nn.Tanh())

        
    def forward(self, x):
        x2 = F.interpolate(x, scale_factor=0.5,mode='bilinear')
        x3 = F.interpolate(x2, scale_factor=0.5,mode='bilinear')
        
        out3 = self.conv_in3( x3 )
        out3 = self.iblock1_3(out3)
        out3 = self.iblock2_3(out3) 
        out3 = self.iblock3_3(out3) 
        out3 = self.iblock4_3(out3)
        out3 = self.iblock5_3(out3) 
        out3 = self.iblock6_3(out3) 
        y3 = self.conv_out3(out3)
        
        
        out2 = self.fusion2(out3,x2,y3)
        out2 = self.iblock1_2(out2)
        out2 = self.iblock2_2(out2) 
        out2 = self.iblock3_2(out2) 
        out2 = self.iblock4_2(out2)
        out2 = self.iblock5_2(out2) 
        y2 = self.conv_out2(out2)
        
        out1 = self.fusion1(out2,x,y2)
        out1 = self.iblock1(out1)
        out1 = self.iblock2(out1) 
        out1 = self.iblock3(out1) 
        out1 = self.iblock4(out1)
        out1 = self.conv_out( out1  )

        
        return out1,y2,y3

        
        
