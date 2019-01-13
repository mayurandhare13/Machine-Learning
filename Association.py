# APRIORI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from 

dataset = pd.read_csv('data/Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# training Apriori on dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# visualization
res = list(rules)

'''
other recommendation system techniques -> collaborative filtering, user profile to add additional info.
More Advance ->  Neighborhood Model, Latent Factor Model, 
'''