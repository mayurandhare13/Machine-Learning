library(e1071)
library(ggplot2)
library(randomForest)
library(caTools)

set.seed(139)

## Naive Bayes
dataset = read.csv('data/Social_Network_Ads.csv')

# encoding categorical features
dataset$Gender = factor(dataset$Gender, levels = c('Male', 'Female'), labels = c(1, 2))
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1)) # NB won't work unless it has y as factors

dataset = dataset[, 2:5]

# split data
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
train_data = subset(dataset, split == TRUE)
test_data = subset(dataset, split == FALSE)

# Scaling
train_data[, 2:3] = scale(train_data[, 2:3])
test_data[, 2:3] = scale(test_data[, 2:3])

nb_classifier = naiveBayes(x = train_data[, -4], y = train_data$Purchased)

y_pred = predict(nb_classifier, newdata=test_data[-4])

cm = table(test_data[, 4], y_pred)