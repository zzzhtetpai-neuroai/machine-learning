import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset


def make_train_step_fn(model,loss_fn,optimizer):
    def perform_train_step(x,y):
        model.train()
        yhat=model(x)
        loss=loss_fn(yhat,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return perform_train_step# no parentheses here if use here then it will call the function 


def make_val_step_fn(model,loss_fn):
    def perform_val_step(x,y):
        model.eval()
        yhat=model(x)
        loss=loss_fn(yhat,y)
        return loss.item()
    return perform_val_step


def minibatch(device,data_loader,step_fn):
    minibatch_losses=[]
    for x_batch,y_batch in data_loader:
        x_batch,y_batch=x_batch.to(device),y_batch.to(device)
        minibatch_loss=step_fn(x_batch,y_batch)
        minibatch_losses.append(minibatch_loss)
    loss=np.mean(minibatch_losses)
    return loss


# Runs only when this file is executed directly, not when other scripts import the functions above
if __name__=="__main__":
    true_b=1
    true_w=2
    N=100
    np.random.seed(42)
    x=np.random.rand(N,1)
    y=true_b+true_w*x+(.1*np.random.randn(N,1))

    idx=np.arange(N)
    np.random.shuffle(idx)
    train_idx=idx[:int(0.8*N)]
    val_idx=idx[int(0.8*N):]
    x_train,y_train=x[train_idx],y[train_idx]
    x_val,y_val=x[val_idx],y[val_idx]

    device='cuda' if torch.cuda.is_available() else 'cpu'
    x_train_tensor=torch.as_tensor(x_train,dtype=torch.float32)
    y_train_tensor=torch.as_tensor(y_train,dtype=torch.float32)
    train_data=TensorDataset(x_train_tensor,y_train_tensor)
    train_loader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
    # next(iter(train_loader)) this is what a data laoder looks like

    lr=0.1
    torch.manual_seed(42)
    model=nn.Sequential(nn.Linear(1,1)).to(device)
    optimizer=optim.SGD(model.parameters(),lr=lr)
    loss_fn=nn.MSELoss(reduction='mean')

    train_step_fn=make_train_step_fn(model,loss_fn,optimizer)
    n_epochs=1000
    losses=[]
    for epoch in range(n_epochs):
        loss=minibatch(device,train_loader,train_step_fn)
        losses.append(loss)
    print(model.state_dict())   # should be close to the true values b=1, w=2
