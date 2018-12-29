library(e1071)
library(ggplot2)

dataset = read.csv('data/Position_Salaries.csv')
dataset = dataset[2:3]

sv_regressor = svm(formula=Salary ~ ., data=dataset, type='eps-regression')

y_pred = predict(sv_regressor, data.frame(Level = 6.5))

ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), colors='red') +
  geom_line(aes(x=dataset$Level, y=predict(sv_regressor, newdata=dataset)), colors='blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab("Levels") + 
  ylab("Salary")