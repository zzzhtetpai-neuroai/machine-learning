import numpy as np
import matplotlib.pyplot as plt
import random

#Set the parameters

N=10000 #Number of customers
d=9     #Number of marketing strategies
#Define success rates for each strategy
conversion_rates=[0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.01]

#Create a simulation matrix
X=np.array(np.zeros((N,d)))

#Defining winning condition
for i in range(N):
    for j in range(d):
        if np.random.rand()<conversion_rates[j]:
            X[i][j]=1

#Prepare the necesary variables for comparison
strategies_selected_rs=[]
strategies_selected_ts=[]
total_reward_rs=0
total_reward_ts=0
number_of_rewards_1=[0]*d
number_of_rewards_0=[0]*d


for n in range(0,N):
    #Deploy Random Selection Strategy
    strategy_rs=random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs=X[n][strategy_rs]
    total_reward_rs=total_reward_rs+reward_rs

    #Deploy Thompson Sampling Strategy
    selected_ts=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            selected_ts=i
    reward_ts=X[n][selected_ts]   
    if reward_ts==1:
        number_of_rewards_1[selected_ts]+=1
    else:
        number_of_rewards_0[selected_ts]+=1
    strategies_selected_ts.append(selected_ts)
    total_reward_ts+=reward_ts


#Comparing the results(computing relative return)
relative_return=((total_reward_ts-total_reward_rs)/total_reward_rs)*100
print(f"Relative return: {relative_return:.0f}%")

#Visualizing the results
plt.hist(strategies_selected_ts)
plt.title("Histogram of strategies selected by Thompson Sampling")
plt.xlabel("Strategy")
plt.ylabel("Number of times each strategy was selected")
plt.show()

