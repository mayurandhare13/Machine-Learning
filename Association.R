## APRIORI

library(arules)

# Dataset
# dataset should be sparse matrix to operate
dataset <- read.transactions("data/Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Train
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualization
inspect(sort(rules, by = 'lift')[1:10])