import numpy as np

#setup 
N=10000         
conversion_rates=[0.15,0.04,0.13,0.11,0.05]
d=len(conversion_rates)
X=np.zeros((N,d))
npos_reward=np.zeros(d)
nneg_reward=np.zeros(d)

#define winning condition
for i in range(N):
    for j in range(d):
        if np.random.rand()<conversion_rates[j]:
            X[i][j]=1

#choose the best slot machine
for i in range(N):
    selected=0
    max_random=0
    for j in range(d):
        random_beta=np.random.beta(npos_reward[j]+1,nneg_reward[j]+1)
        if random_beta>max_random:
            max_random=random_beta
            selected=j
    if X[i][selected]==1:
        npos_reward[selected]+=1    
    else:
        nneg_reward[selected]+=1

nselected=npos_reward+nneg_reward
for i in range(d):
    print(f"Machine {i+1} was selected {nselected[i]} times")
print(f"Conclusion: Machine {np.argmax(nselected) + 1} is the best choice.")

