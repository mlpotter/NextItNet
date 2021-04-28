# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:39:11 2021

@author: lpott
"""
import argparse
from torch.utils.data import DataLoader
import torch

from preprocessing import *
from dataset import *
from metrics import *
from model import *

from time import time

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--read_filename',type=str,help='The filename to read all the MovieLens-1 million data from to the Dataframe',default="ml-1m\\ratings.dat")
parser.add_argument('--hitsat',type=int,help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)',default=10)
parser.add_argument('--max_len',type=int,help='Maximum length for the sequence',default=200)
parser.add_argument('--min_len',type=int,help="Minimum session length for a sequence (filter out sessions less than this",default=10)


# ----------------- Variables ----------------------#


args = parser.parse_args()

read_filename = args.read_filename
k = args.hitsat
max_length = args.max_len
min_len = args.min_len


# ------------------Data Initialization----------------------#

ml_1m = create_df(read_filename)

ml_1m = filter_df(ml_1m)

reset_object = reset_df()
ml_1m = reset_object.fit_transform(ml_1m)

n_users,n_items,n_ratings,n_timestamp = ml_1m.nunique()

user_history = create_user_history(ml_1m)
user_noclicks = create_user_noclick(user_history,ml_1m,n_items)

train_history,val_history,test_history = train_val_test_split(user_history,max_length=max_length)

pad_token = n_items

train_dataset = GRUDataset(train_history,mode='train',max_length=max_length,pad_token=pad_token)
val_dataset = GRUDataset(val_history,mode='eval',max_length=max_length,pad_token=pad_token)
test_dataset = GRUDataset(test_history,mode='eval',max_length=max_length,pad_token=pad_token)

output_dim = n_items

val_dl = DataLoader(val_dataset,batch_size=64)
test_dl = DataLoader(test_dataset,batch_size=64)
# ------------------ Metric / Objective Init ------------------#
loss_fn = nn.CrossEntropyLoss(ignore_index=n_items)
Recall_Object = Recall(user_noclicks,n_users,n_items,k=k)
# ==================Hyper-parameter search =================== #
num_epochs_grid= [10]
lr_grid = [1e-2,1e-3, 5-3] 
batch_size_grid = [32,64,128]
reg_grid = [0,1e-5]

embedding_dim_grid = [32,64,128]
hidden_layers_grid = [1,2]
dilations_grid = [[1,2,4],[1,2,4,8],[1,2,4,16]]

search_file = open("grid_search{:d}.txt".format(k),"w")
time_start = time()
search_file.write("Epoch,lr,batch_size,reg,embedding_dim,hidden_layers,dilations"+"\n")
for num_epochs in num_epochs_grid:
    for lr in lr_grid:
        for batch_size in batch_size_grid:
            for reg in reg_grid:
                for embedding_dim in embedding_dim_grid:
                    for hidden_layers in hidden_layers_grid:
                        for dilations in dilations_grid:
                            
                            time_optimize=time()
                            # ------------------ Train dataloader ------------------------ #
                            train_dl = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
                            # ------------------Model Initialization---------------------- #
                            model = NextItNet(embedding_dim,output_dim,hidden_layers=hidden_layers,dilations=dilations,pad_token=n_items,max_len=max_length).cuda()
                            optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)
                            # ------------------Training Initialization----------------------#
                            
                            max_train_hit = 0
                            max_val_hit = 0
                            max_test_hit = 0
                            print(num_epochs,lr,batch_size,embedding_dim,hidden_layers,dilations)
                            for epoch in range(num_epochs):
                                
                                model.train()  
                                
                                running_loss = 0
                            
                                for data in train_dl:
                                    optimizer.zero_grad()
                            
                                    inputs,labels,x_lens,uid = data
                                    
                                    outputs = model(inputs.cuda())
                            
                                    loss = loss_fn(outputs.view(-1,outputs.size(-1)),labels.view(-1).cuda())
                                    loss.backward()
                                    optimizer.step()
                                    
                                    running_loss += loss.detach().cpu().item()
                            
                                training_hit = Recall_Object(model,train_dl)
                                validation_hit = Recall_Object(model,val_dl)
                                testing_hit = Recall_Object(model,test_dl)
                                
                                if max_val_hit < validation_hit:
                                    max_val_hit = validation_hit
                                    max_test_hit = testing_hit
                                    max_train_hit = training_hit
                            
                            time_optimize = time()-time_optimize
                            print("Search Time: {:.2f}".format(time_optimize))
                                
                            search_file.write("="*100+"\n")
                            search_file.write("{:f},{:f},{:f},{:f},{:f},{:f},{:s}".format(num_epochs,lr,batch_size,reg,embedding_dim,hidden_layers,"-".join(list(map(str,dilations)))))
                            search_file.write("\n")
                            search_file.write("{:.2f},{:.2f},{:.2f}".format(max_train_hit,max_val_hit,max_test_hit))
                            search_file.write("\n")

                            torch.cuda.empty_cache()
time_end = time()
total_time = time_end - time_start

print("Hyperparameter Search Time: {:.2f}".format(total_time))
search_file.close()