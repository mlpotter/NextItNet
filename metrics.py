# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:13 2021

@author: lpott
"""
import torch
import numpy as np

class Recall(object):
    def __init__(self,user_noclick,n_users,n_items,k=10):
        print("="*10,"Creating Hit@{:d} Metric Object".format(k),"="*10)

        self.user_noclick = user_noclick
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
                

    def __call__(self,model,dataloader):
        
        model.eval()
        with torch.no_grad():
            
            total_hits = 0 
            for data in dataloader:
                inputs,labels,x_lens,uid = data
                outputs = model(inputs.cuda())
                                
                for i,uid in enumerate(uid.squeeze()):
                    negatives,probabilities = self.user_noclick[uid.item()]
                    sampled_negatives = np.random.choice(negatives,size=100,replace=False,p=probabilities).tolist() + [labels[i,x_lens[i].item()-1].item()]
                                        
                    topk_items = outputs[i,x_lens[i].item()-1,sampled_negatives].argsort(0,descending=True)[:self.k]
                    total_hits += torch.sum(topk_items == 100).cpu().item()       
                                        
                                                        
           
                
        return total_hits/self.n_users*100