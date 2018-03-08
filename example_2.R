library(Matrix)
library(caret)
library(xgboost)
library(ROCR)

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
train_df <- read.csv(url, header = FALSE, na.strings = " ?")

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"
decrp <- readLines(url)
names <- sub("(.*):.*", "\\1", decrp[97:110])
names
colnames(train_df) <- c(names, "Y")
head(train_df)
str(train_df)

## one-hot encoding
encoding <- dummyVars(~ . - Y, train_df)
train_data <- predict(encoding, train_df, na.action = na.pass)
train_data <- as(train_data, "sparseMatrix") 

train_label <- as.numeric(train_df$Y) - 1
train <- xgb.DMatrix(data = train_data, label = train_label)

## test data 
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
test_df <- read.csv(url, header = FALSE, na.strings = " ?", skip = 1)
colnames(test_df) <- c(names, "Y")
encoding <- dummyVars(~ . - Y, test_df)
test_data <- predict(encoding, test_df, na.action = na.pass)
test_data <- as(test_data, "sparseMatrix") 
test_label <- as.numeric(test_df$Y) - 1
test <- xgb.DMatrix(data = test_data, label = test_label)


param_1 <- list(
  eta = 0.3,             # learning rate
  gamma = 0,             # minimum reduction of loss for split
  max.depth = 6,         # maximum depth of the tree
  subsample = 1,         # row subsampling
  colsample_bylevel = 1, # feature subsampling
  lambda = 1             # l2 penalty
)

## 5-fold cross validation
gbcv_1 <- xgb.cv(params = param_1, data = train, nfold = 5, 
                 objective = "binary:logistic", nrounds = 200, 
                 early_stopping_rounds = 15, eval_metric = "auc", 
                 missing = NA, seed = 12345)

param_2 <- list(
  eta = 0.15,              # learning rate
  gamma = 0,               # minimum reduction of loss for split
  max.depth = 4,           # maximum depth of the tree
  subsample = 0.7,         # row subsampling
  colsample_bylevel = 0.5, # feature subsampling
  lambda = 3
)

gbcv_2 <- xgb.cv(params = param_2, data = train, nfold = 5, 
                 objective = "binary:logistic", nrounds = 500, 
                 early_stopping_rounds = 10, eval_metric = "auc", 
                 missing = NA, seed = 12345)


## validation set
set.seed(12345)
index <- sample(1:nrow(train_df), 6000, replace = FALSE)
train_df_sub <- train_df[-index,]
vali_df <- train_df[index,]

encoding <- dummyVars(~ . - Y, train_df_sub)
train_sub_data <- predict(encoding, train_df_sub, na.action = na.pass)
train_sub_data <- as(train_sub_data, "sparseMatrix") 

train_sub_label <- as.numeric(train_df_sub$Y) - 1
train_sub <- xgb.DMatrix(data = train_sub_data, label = train_sub_label)

encoding <- dummyVars(~ . - Y, vali_df)
vali_data <- predict(encoding, vali_df, na.action = na.pass)
vali_data <- as(vali_data, "sparseMatrix") 

vali_label <- as.numeric(vali_df$Y) - 1
vali <- xgb.DMatrix(data = vali_data, label = vali_label)



watchlist <- list(train = train_sub, test = vali)
xgb_vali_1 <- xgb.train(params = param_1, data = train_sub, 
                        watchlist = watchlist, 
                        objective = "binary:logistic", 
                        nrounds = 200, early_stopping_rounds = 15, 
                        eval_metric = "auc", missing = NA, seed = 12345)

xgb_vali_2 <- xgb.train(params = param_2, data = train_sub, 
                        watchlist = watchlist, 
                        objective = "binary:logistic", 
                        nrounds = 200, early_stopping_rounds = 15, 
                        eval_metric = "auc", missing = NA, seed = 12345)




## Final model

xgbtree_1 <- xgboost(params = param_1, data = train, verbose = 0, 
                     objective = "binary:logistic", nrounds = 25, 
                     missing = NA, seed = 12345)

xgbtree_2 <- xgboost(params = param_2, data = train, verbose = 0, 
                     objective = "binary:logistic", nrounds = 130, 
                     missing = NA, seed = 12345)


performance(prediction(predict(xgbtree_1, test), test_label), "auc")
performance(prediction(predict(xgbtree_2, test), test_label), "auc")

importance <- xgb.importance(feature_names = dimnames(train)[[2]], 
                             model = xgbtree_1)
xgb.ggplot.importance(importance, top_n = 15) + theme(legend.position="none")
