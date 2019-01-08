library(e1071)
library(ggplot2)
library(rpart) 
library(randomForest)
library(caTools)

set.seed(139)

dataset = read.csv('data/Position_Salaries.csv')
dataset = dataset[2:3]

sv_regressor = svm(formula=Salary ~ ., data=dataset, type='eps-regression')

y_pred = predict(sv_regressor, data.frame(Level = 6.5))

ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), colors='red') +
  geom_line(aes(x=dataset$Level, y=predict(sv_regressor, newdata=dataset)), colour='blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab("Levels") + 
  ylab("Salary")

## Polynomial Regression

dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary~., data=dataset)

y_pred = predict(poly_reg, newdata = data.frame(Level=6.5, Level2=6.5^2,
                                                Level3=6.5^3, Level4=6.5^4))

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), colour='red') +
  geom_line(aes(x=x_grid, y=predict(poly_reg, newdata=data.frame(Level=x_grid,
                                                                        Level2=x_grid^2,
                                                                        Level3=x_grid^3,
                                                                        Level4=x_grid^4))), 
            colour='blue') +
  ggtitle("Truth or Bluff (Poly Regression)") +
  xlab("Levels") + 
  ylab("Salary")

## Decision Tree Regressor

  dt_regressor = rpart(formula=Salary~., data=dataset, 
                      control=rpart.control(minsplit=1))

  y_pred = predict(dt_regressor, data.frame(Level=6.5))

  x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
  
  ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), colour='red') +
  geom_line(aes(x=x_grid, y=predict(dt_regressor, 
                            newdata=data.frame(Level=x_grid))), 
            colour='blue') +
  ggtitle("Truth or Bluff (Decision Tree Regression)") +
  xlab("Levels") + 
  ylab("Salary")


## Random Forest Regressor

  rf_regressor = randomForest(x=dataset[1],
                              y=dataset$Salary, ntree=200)
  # dataset$Salary -> returns Series
  # dataset[1] -> returns dataframe

  y_pred = predict(rf_regressor, data.frame(Level=6.5))

  x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
  
  ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), colour='red') +
  geom_line(aes(x=x_grid, y=predict(rf_regressor, 
                            newdata=data.frame(Level=x_grid))), 
            colour='blue') +
  ggtitle("Truth or Bluff (Random Forest Regression)") +
  xlab("Levels") + 
  ylab("Salary")


## Logistic Regression
dataset = read.csv('data/Social_Network_Ads.csv')

# encoding categorical features
dataset$Gender = factor(dataset$Gender, levels = c('Male', 'Female'), labels = c(1, 2))

dataset = dataset[, 2:5]

# split data
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
train_data = subset(dataset, split == TRUE)
test_data = subset(dataset, split == FALSE)

# Scaling
train_data[, 2:3] = scale(train_data[, 2:3])
test_data[, 2:3] = scale(test_data[, 2:3])

logi_regressor = glm(formula = Purchased ~ . , family = binomial,
                     data = train_data)

y_prob = predict(logi_regressor, type='response', newdata=test_data[-4])
y_pred = ifelse(y_prob > 0.5, 1, 0)

cm = table(test_data[, 4], y_pred)
