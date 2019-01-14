# Reinforcement Learning
# Upper Confidence Bound

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

# UCB
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d): 
        if(numbers_of_selections[i] > 0):
            avg_reward = sums_of_rewards[i] / numbers_of_selections[i] # avg reward of adv i upto round n
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = delta_i + avg_reward
        else:
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward 
    total_reward += reward

print(total_reward)

# Visualization
plt.hist(ads_selected)
plt.title("Histogram of ads selected")
plt.xlabel("Ads")
plt.ylabel("Number of times ads selected")
plt.show()