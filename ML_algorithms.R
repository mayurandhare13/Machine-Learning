library(e1071)
library(ggplot2)
library(randomForest)
library(caTools)

set.seed(139)


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

## Naive Bayes
nb_classifier = naiveBayes(x = train_data[, -4], y = train_data$Purchased)

y_pred = predict(nb_classifier, newdata=test_data[-4])

cm = table(test_data[, 4], y_pred)

## Decision Tree
# we don't need to scale features as RF or DT are not based not euclidean distances
# but based on conditions of independant variable

library(rpart)

dt_classifier = rpart(formula = Purchased ~ ., data=train_data)

y_pred = predict(dt_classifier, newdata=test_data[-4], type='class')
# type='class' will give you predicted value(here, 0/1). w/o it returns probability

cm = table(test_data[, 4], y_pred)

## Random Forest Classifier
rf_classifier = randomForest(x=train_data[-4],
                            y=train_data$Purchased, ntree=20)

y_pred = predict(rf_classifier, newdata=test_data[-4])

cm = table(test_data[, 4], y_pred)

y_pred_2 = predict(rf_classifier, newdata = train_data[-4])

cm = table(train_data[, 4], y_pred_2)