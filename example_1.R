library(xgboost)
library(dplyr)
library(ggplot2)
library(DiagrammeR)
library(Ckmeans.1d.dp)


df <- iris
str(df)

X <- as.matrix(df[,-5])
Y <- as.numeric(df[,5]) - 1
dtrain <- xgb.DMatrix(data = X, label = Y)


xgbtree <- xgboost(data = dtrain, objective = "multi:softmax", 
                   nrounds = 20, max.depth = 3, num_class = 3, eta = 0.2)
pred <- predict(xgbtree, X)
head(pred)

xgbtree <- xgboost(data = dtrain, objective = "multi:softprob", 
                   nrounds = 20, max.depth = 3, num_class = 3, eta = 0.2)
pred <- predict(xgbtree, X)
head(matrix(pred, ncol = 3, byrow = TRUE))

model <- xgb.dump(xgbtree, with_stats = TRUE)   # get tree structure
model[1:14]
xgb.plot.tree(feature_names = dimnames(dtrain)[[2]], model = xgbtree, 
              n_first_tree = 1)
importance <- xgb.importance(feature_names = dimnames(dtrain)[[2]], 
                             model = xgbtree)
xgb.ggplot.importance(importance) + theme(legend.position="none")




