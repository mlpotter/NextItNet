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

parser = argparse.ArgumentParser()

parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=25)
parser.add_argument('--alpha', type=float, help='Learning Rate', default=0.001)
parser.add_argument('--embedding_dim',type=int,help="Size of item embedding",default=64)
parser.add_argument('--read_filename',type=str,help='The filename to read all the MovieLens-1 million data from to the Dataframe',default="ml-1m\\ratings.dat")
parser.add_argument('--batch_size',type=int,help='The batch size for stochastic gradient descent',default=32)
parser.add_argument('--reg',type=float,help='The regularization strength on l2 norm',default = 0.0)
parser.add_argument('--hidden_layers',type=int,help="The number of hidden layeres",default=2)
parser.add_argument('--dilations',type=str,help="The dilation scheme of the hidden layers",default="1,2,4,8")
parser.add_argument('--hitsat',type=int,help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)',default=10)
parser.add_argument('--max_len',type=int,help='Maximum length for the sequence',default=200)


# ----------------- Variables ----------------------#


args = parser.parse_args()

read_filename = args.read_filename

num_epochs = args.num_epochs
lr = args.alpha
batch_size = args.batch_size
reg = args.reg

embedding_dim = args.embedding_dim
hidden_layers = args.hidden_layers
dilations= list(map(int,args.dilations.split(",")))

k = args.hitsat
max_length = args.max_len


# ------------------Data Initialization----------------------#

ml_1m = create_df(read_filename)

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

train_dl = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
val_dl = DataLoader(val_dataset,batch_size=64)
test_dl = DataLoader(test_dataset,batch_size=64)
# ------------------Model Initialization----------------------#

model = NextItNet(embedding_dim,output_dim,hidden_layers=hidden_layers,dilations=dilations,pad_token=n_items,max_len=max_length).cuda()
loss_fn = nn.CrossEntropyLoss(ignore_index=n_items)
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)
Recall_Object = Recall(user_noclicks,n_users,n_items,k=k)
# ------------------Training Initialization----------------------#

max_train_hit = 0
max_val_hit = 0
max_test_hit = 0

for epoch in range(num_epochs):
    print("="*20,"Epoch {}".format(epoch+1),"="*20)
    
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
    

    print("Training CE Loss: {:.5f}".format(running_loss/len(train_dl)))
    print("Training Hits@{:d}: {:.2f}".format(k,training_hit))
    print("Validation Hits@{:d}: {:.2f}".format(k,validation_hit))
    print("Testing Hits@{:d}: {:.2f}".format(k,testing_hit))
    

print("="*100)
print("Maximum Training Hit@{:d}: {:.2f}".format(k,max_train_hit))
print("Maximum Validation Hit@{:d}: {:.2f}".format(k,max_val_hit))
print("Maximum Testing Hit@{:d}: {:.2f}".format(k,max_test_hit))
