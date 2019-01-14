# Reinforcement Learning
# Thompson Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

dataset = pd.read_csv("data/Ads_CTR_Optimisation.csv")

N = 10000   # total number of times advs flashed
d = 10      # Number of adv versions

# Random Selection
ads_selected = []
total_reward = 0
for i in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[i, ad]
    total_reward += reward

print("Random Sampling:- ", total_reward)


# Thompson Algorithm
number_of_reward_0 = [0] * d
number_of_reward_1 = [0] * d
ads_selected = []
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_reward_1[i]+1, number_of_reward_0[i]+1) 
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if(reward == 1):
        number_of_reward_1[ad] += 1
    else:
        number_of_reward_0[ad] += 1
    total_reward += reward

print("Thompson Sampling:- ", total_reward)

# Visualization
plt.hist(ads_selected)
plt.title("Histogram of ads selected [Thompson Sampling]")
plt.xlabel("Ads")
plt.ylabel("Number of times ads selected")
plt.show()