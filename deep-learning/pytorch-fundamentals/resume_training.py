import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,random_split
from training_loop import make_train_step_fn,make_val_step_fn,minibatch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

true_b=1
true_w=2
N=100

np.random.seed(42)
x=np.random.rand(N,1)
y=true_b+true_w*x+(.1*np.random.randn(N,1))

device='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(13)
x_tensor=torch.as_tensor(x).float()
y_tensor=torch.as_tensor(y).float()
dataset=TensorDataset(x_tensor,y_tensor)

ratio=.8
n_total=len(dataset)
n_train=int(n_total*ratio)
n_val=n_total-n_train
train_data,val_data=random_split(dataset,[n_train,n_val])
train_loader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
val_laoder=DataLoader(dataset=val_data,batch_size=16,shuffle=False)

lr=0.1
model=nn.Sequential(nn.Linear(1,1)).to(device)
optimizer=optim.SGD(model.parameters(),lr=lr)
loss_fn=nn.MSELoss(reduction='mean')
train_step_fn=make_train_step_fn(model,loss_fn,optimizer)
val_step_fn=make_val_step_fn(model,loss_fn)

writer=SummaryWriter('logs//loaded_model_experiment_1')
x_dummy, y_dummy = next(iter(train_loader)) 
writer.add_graph(model, x_dummy.to(device))

checkpoint=torch.load('model_checkpoint.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
saved_n_epochs=checkpoint['n_epochs']
saved_losses=checkpoint['train_losses']
saved_val_losses=checkpoint['val_losses']
model.train()
n_epochs=200
losses=saved_losses.copy()
val_losses=saved_val_losses.copy()

for epoch in range(n_epochs):
    loss=minibatch(device,train_loader,train_step_fn)
    losses.append(loss)
    with torch.no_grad():
        val_loss=minibatch(device,val_laoder,val_step_fn)
        val_losses.append(val_loss)
    
    writer .add_scalars('loss',{'train loss':loss,'val_loss':val_loss},saved_n_epochs+epoch)

writer.close()