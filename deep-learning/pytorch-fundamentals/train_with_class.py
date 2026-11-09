import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from step_by_step import StepByStep


true_b=1
true_w=2
N=100

np.random.seed(42)
x=np.random.rand(N,1)
y=true_b+true_w*x+(.1*np.random.randn(N,1))

torch.manual_seed(42)
x_tensor=torch.as_tensor(x).float()
y_tensor=torch.as_tensor(y).float()
dataset=TensorDataset(x_tensor,y_tensor)

ratio=.8
n_total=len(dataset)
n_train=int(n_total*ratio)
n_val=n_total-n_train
train_data,val_data=random_split(dataset,[n_train,n_val])
train_loader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
val_loader=DataLoader(dataset=val_data,batch_size=16,shuffle=True)

lr=0.1
model=nn.Sequential(nn.Linear(1,1))
optimizer=optim.SGD(model.parameters(),lr=lr)
loss_fn=nn.MSELoss(reduction='mean')

n_epochs=200
sbs=StepByStep(model,loss_fn,optimizer)
sbs.set_loaders(train_loader,val_loader)
sbs.set_tensorboard('classy')
sbs.train(n_epochs=n_epochs)
print(model.state_dict())

