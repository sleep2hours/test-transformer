import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt


def clones(module,N):
    #把一个网络层拷贝N份
    return nn.ModuleList([copy.deepcopy(module)for i in range(N)])

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

class Encoder(nn.Module):     #整个编码器结构
    def __init__(self,layer,N):
        super().__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)


class SubLayer(nn.Module):    #编码器的每个重复单元
    def __init__(self,size,dropout):
        super(SubLayer, self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=dropout

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderLayers(nn.Module):    #多头注意力+全连接
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayers, self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SubLayer(size,dropout),2)
        self.size=size

    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder, self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)

class DecoderLayers(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayers, self).__init__()
        self.size=size
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.src_attn=src_attn
        self.sublayers=clones(SubLayer(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        m=memory
        x=self.sublayers[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayers[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayers[2](x,self.feed_forward)

def subsequent_mask(size):                   #制造mask
    attn_shape=(1,size,size)
    mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(mask)==0

def attention(query,key,value,mask=None,dropout=None):
    dk=query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(dk)
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):    #h为头数，d_model为模型输入维数，d_model要可以被h整除
        super(MultiHeadAttention, self).__init__()
        assert d_model%h==0
        self.d_k=d_model//h
        self.h=h
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not  None:
            mask=mask.unsqueeze(1)
        nbatches=query.size(0)

        query,key,value=[l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
        x,self.attn=attention(query,key,value,mask,dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)







