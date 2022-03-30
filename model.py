import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt


class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.generator=generator

    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator, self).__init__()
        self.proj=nn.Linear(d_model,vocab)

    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)




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

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(dropout)

        #计算位置编码
        pos=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(-math.log(10000)*torch.arange(0,d_model,2)/d_model)
        pos[:,0::2]=torch.sin(position*div_term)
        pos[:,1::2]=torch.cos(position*div_term)
        pos=pos.unsqueeze(0)
        self.register_buffer('pos',pos)

    def forward(self,x):
        x=x+Variable(self.pos[:,:x.size(1),:x.size(2)],requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    c=copy.deepcopy
    attn=MultiHeadAttention(h,d_model)
    ff=FeedForward(d_model,d_ff,dropout)
    position=PositionalEncoding(d_model,dropout)
    model=EncoderDecoder(
        Encoder(EncoderLayers(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayers(d_model,c(attn),c(attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model









