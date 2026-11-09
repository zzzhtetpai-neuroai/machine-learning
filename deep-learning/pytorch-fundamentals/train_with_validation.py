import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from training_loop import make_train_step_fn,make_val_step_fn,minibatch

true_b=1
true_w=2
N=100


np.random.seed(42)
x=np.random.rand(N,1)
y=true_b+true_w*x+(.1*np.random.randn(N,1))
device='cuda' if torch.cuda.is_available() else 'cpu'
x_train_tensor=torch.as_tensor(x,dtype=torch.float32) #.to(device)
y_train_tensor=torch.as_tensor(y,dtype=torch.float32) #.to(device)

dataset=TensorDataset(x_train_tensor,y_train_tensor)
ratio=.8
n_total=len(dataset)
n_train=int(n_total*ratio)
n_val=n_total-n_train
train_data,val_data=random_split(dataset,[n_train,n_val])
train_loader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
val_loader=DataLoader(dataset=val_data,batch_size=16,shuffle=False)

lr=0.1
torch.manual_seed(42)
model=nn.Sequential(nn.Linear(1,1)).to(device)
optimizer=optim.SGD(model.parameters(),lr=lr)
loss_fn=nn.MSELoss(reduction='mean')
train_step_fn=make_train_step_fn(model,loss_fn,optimizer)
val_step_fn=make_val_step_fn(model,loss_fn)

writer=SummaryWriter('logs/linear_regression_experiment_1')

n_epochs=200
losses=[]
val_losses=[]
for epoch in range(n_epochs):
    loss=minibatch(device,train_loader,train_step_fn)
    losses.append(loss)
    with torch.no_grad():
        val_loss=minibatch(device,val_loader,val_step_fn)
        val_losses.append(val_loss)
    writer.add_scalars('loss',{'train loss':loss,'val loss':val_loss},epoch)
writer.close()

checkpoint={'n_epochs':n_epochs,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'train_losses':losses,
            'val_losses':val_losses}
torch.save(checkpoint,'model_checkpoint.pth')

print(model.state_dict())

# plt.figure(figsize=(10,6))
# plt.plot(losses,label='Train loss')
# plt.plot(val_losses,label='Val loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title("Training and Validation loss")
# plt.legend()
# plt.show()




