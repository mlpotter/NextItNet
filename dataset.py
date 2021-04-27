# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:41:47 2021

@author: lpott
"""

import torch
from torch.utils.data import Dataset

class GRUDataset(Dataset):
    def __init__(self, u2seq, mode='train',max_length=200,pad_token=0):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_length
        self.pad_token = pad_token
        self.mode=mode

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        
        
        if self.mode == 'train':
            tokens = seq[:-1]
            labels = seq[1:]

            x_len = len(tokens)
            y_len = len(labels)

            x_mask_len = self.max_len - x_len
            y_mask_len = self.max_len - y_len

            tokens =  tokens + [self.pad_token] * x_mask_len 
            labels =  labels + [self.pad_token] * y_mask_len

            
        if self.mode == 'eval':
            tokens = seq[:-1]
            labels = seq[-1:]
            
            x_len = len(tokens)
            
            labels = [self.pad_token] * (x_len-1) + labels
            
            y_len = len(labels)


            x_mask_len = self.max_len - x_len
            y_mask_len = self.max_len - y_len

            tokens =  tokens + [self.pad_token] * x_mask_len 
            labels =  labels + [self.pad_token] * y_mask_len
        
        return torch.LongTensor(tokens), torch.LongTensor(labels),torch.LongTensor([x_len]),torch.LongTensor([user])

    def _getseq(self, user):
        return self.u2seq[user]