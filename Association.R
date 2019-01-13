library(arules)

## APRIORI
# people bought this also bought that.

# Dataset
# dataset should be sparse matrix to operate
dataset <- read.transactions("data/Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Train apriori
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualization
inspect(sort(rules, by = 'lift')[1:10])

## ECLAT
# items frequently bought together

# Train eclat
sets = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

# Visualization
inspect(sort(sets, by = 'support')[1:10])
