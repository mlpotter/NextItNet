# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:39:46 2021

@author: lpott
"""
import numpy as np
import pandas as pd
import os
from time import time

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder 

def create_df(filename=None):
    print("="*10,"Creating DataFrame","="*10)
    df = pd.read_csv(filename,sep='::',header=None)
    df.columns= ['user_id','item_id','rating','timestamp']
    df.sort_values('timestamp',inplace=True)
    
    
    print(df.nunique())
    print(df.shape)
    
    return df.reset_index(drop=True)

class reset_df(object):
    
    def __init__(self):
        print("="*10,"Initialize Reset DataFrame Object","="*10)
        self.item_enc = LabelEncoder()
        self.user_enc = LabelEncoder()
        
    def fit_transform(self,df):
        print("="*10,"Resetting user ids and item ids in DataFrame","="*10)
        df['item_id'] = self.item_enc.fit_transform(df['item_id'])
        df['user_id'] = self.user_enc.fit_transform(df['user_id'])
        
        assert df.user_id.min() == 0
        assert df.item_id.min() == 0 
        
        return df
    
    def inverse_transform(self,df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'])
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'])
        return df
    
def create_user_history(df=None):
    if df is None:
        return None
    
    print("="*10,"Creating User Histories","="*10)

    user_history = {}
    for uid in tqdm(df.user_id.unique()):
        sequence= df[df.user_id == uid].item_id.values.tolist()
        if len(sequence) < 10:
            continue
        user_history[uid] = sequence
            
    return user_history

def train_val_test_split(user_history=None,max_length=200):
    if user_history is None:
        return None
    

    print("="*10,"Splitting User Histories into Train, Validation, and Test Splits","="*10)
    train_history = {}
    val_history = {}
    test_history = {}
    for key,history in tqdm(user_history.items(),position=0, leave=True):
        train_history[key] = history[-(max_length+2):-2]
        val_history[key] = history[-(max_length+1):-1]
        test_history[key] = history[(-max_length):]
        
    return train_history,val_history,test_history

def create_user_noclick(user_history,df,n_items):
    print("="*10,"Creating User 'no-click' history","="*10)
    user_noclick = {}
    all_items = np.arange(n_items)

    item_counts = df.groupby('item_id',sort='item_id').size()
    #item_counts = (item_counts/item_counts.sum()).values


    for uid,history in tqdm(user_history.items()):
        no_clicks = list(set.difference(set(all_items.tolist()),set(history)))
        item_counts_subset = item_counts[no_clicks]
        probabilities = ( item_counts_subset/item_counts_subset.sum() ).values

        user_noclick[uid] = (no_clicks,probabilities)
    
    return user_noclick