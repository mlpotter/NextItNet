# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:41 2021

@author: lpott
"""

import torch.nn as nn
import torch.nn.functional as F

class DilatedResBlock(nn.Module):
    def __init__(self,dilation,channel,max_len):
        super(DilatedResBlock,self).__init__()
        self.dilation = dilation
        self.channel = channel
        self.half_channel = int(channel/2)
        self.max_len = max_len
        
        self.reduce = nn.Conv1d(channel,self.half_channel,1)
        self.masked = nn.Conv1d(self.half_channel,self.half_channel,3,dilation=dilation)
        self.increase = nn.Conv1d(self.half_channel,channel,1)
        """
        self.reduce_norm = nn.LayerNorm(normalized_shape=[max_len])#channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=[max_len])#self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=[max_len])#self.half_channel)
        """
        self.reduce_norm = nn.LayerNorm(normalized_shape=channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        
    def forward(self,x):
        y = self.reduce_norm(x.permute(0,2,1)).permute(0,2,1)
        #y = self.reduce_norm(x)

        y = F.leaky_relu(x)
        y = self.reduce(y)
        
                
        y = self.masked_norm(y.permute(0,2,1)).permute(0,2,1)
        y = F.leaky_relu(y)
        y = F.pad(y,pad=(2 + (self.dilation-1)*2,0),mode='constant')
        y = self.masked(y)
      
        
        y = self.increase_norm(y.permute(0,2,1)).permute(0,2,1)
        #y = self.increase_norm(y)
        y = F.leaky_relu(y)
        y = self.increase(y)
        
        return x+y
        

class NextItNet(nn.Module):
    """

    """
    def __init__(self,embedding_dim,output_dim,max_len,hidden_layers=2,dilations=[1,2,4,8],pad_token=0):
        super(NextItNet,self).__init__()
        self.embedding_dim = embedding_dim
        self.channel = embedding_dim
        self.output_dim = output_dim
        self.pad_token = pad_token
        self.max_len = max_len
    
    
        self.item_embedding = nn.Embedding(output_dim+1,embedding_dim,padding_idx=pad_token)
        
        self.hidden_layers = nn.Sequential(*[nn.Sequential(*[DilatedResBlock(d,embedding_dim,max_len) for d in dilations]) for _ in range(hidden_layers)])

        self.final_layer = nn.Linear(embedding_dim, output_dim+1)

    
    def forward(self,x):
        x = self.item_embedding(x).permute(0,2,1)
        x = self.hidden_layers(x)
        x = self.final_layer(x.permute(0,2,1))
        
        return x