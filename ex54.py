# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 09:41:24 2020

@author: Jszhan
"""

import torch
import torch.nn as nn
import torch.optim as optim


datan=torch.load('data_n.t')
datan=datan.to(device='cuda')
target=torch.load('target.t')
target=target.to(device='cuda',dtype=torch.float)
target=torch.unsqueeze(target,1)
n_samples=datan.shape[0]
n_val=int(n_samples*0.2)
shuffle_indexes=torch.randperm(n_samples)
train_indexes=shuffle_indexes[:-n_val]
val_indexes=shuffle_indexes[-n_val:]
train_data=datan[train_indexes]
val_data=datan[val_indexes]
train_target=target[train_indexes]
val_target=target[val_indexes]

#print(datan)
#print(target)

class SubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear=nn.Linear(11,256)
        self.output_linear=nn.Linear(256,1)
    def forward(self,input):
        hidden_t=self.hidden_linear(input)
        activation_t=torch.nn.functional.relu(hidden_t)
        output_t = self.output_linear(activation_t)
        return output_t

sub_model=SubModel().to(device='cuda')

optimizer=optim.SGD(sub_model.parameters(),lr=1e-2)

def trainingloop(epochs,model,optimizer,loss_fn,train_data,train_target,val_data,val_target):
    for epoch in range(1,epochs+1):
        train_target_p=model(train_data)
        train_loss=loss_fn(train_target_p,train_target)
        val_target_p=model(val_data)
        val_loss=loss_fn(val_target_p,val_target)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if epoch<=3 or epoch%100==0:
            print('epoch:{:4},train_loss:{:10.6f},val_loss:{:10.6f}'.format(epoch,train_loss,val_loss))

trainingloop(epochs=3000,
             model=sub_model,
             optimizer=optimizer,
             loss_fn=nn.MSELoss(),
             train_data=train_data,
             train_target=train_target,
             val_data=val_data,
             val_target=val_target)

val_target_p=sub_model(val_data)
correct=0
for i in range(val_target_p.shape[0]):
    target=round(val_target_p[i].item())
    if(target==val_target[i].item()):
        correct+=1
acc=correct/val_target.shape[0]    
    

print("acc:{}".format(acc))