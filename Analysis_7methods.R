# The purpose of this code is to conduct machine lerning on pixel labeling based on hyperspectral intensity values (features)
#
#install.packages("randomForest")
library(randomForest)
#install.packages("e1071")
## below is SVM package
library(e1071)
#for discriminant functional analysis
library(MASS) # for discriminant functional analysis
# below is the package for lda.do for projection(projection)
#install.packages("Rdimtools")
library(Rdimtools)
#install.packages("nnet")
library(nnet) # for multinomial linear regression
# Load caret, for PLS-DA analysis
#install.packages("caret")
library(caret)
#install.packages("pls")
library(pls)
#install.packages("glmnet")
library(glmnet)

# load the data
label=read.table("sorghum_labels.csv")
data=read.table("sorghum_features.csv",sep=",")
# normalize data by dividing 255
norm_data = data/255

# still create the two variables though redundant
x = norm_data; y = label$V1
## make y as a factor and keep numeric y as y.numeric
y.numeric=y; y=as.factor(y)

## five-fold or ten-fold cross-validation. I use predicted.y and observed.y for evaluation
Fold = 5; Plot=FALSE; set.seed(2)
# return the GroupID 7560 values only include 1:5
GroupID = sample(1:Fold,length(y),replace=TRUE) #Components.Bound = 10

predicted.y.matrix=matrix(NA,length(y),7) # return 7560 rows and 7 columns
observed.y=y

# random forest 
for (fold in 1:Fold){
  # fold = 1
  print(c(fold,Fold))
  x.train = x[GroupID!=fold,]
  x.test = x[GroupID==fold,]
  y.train = y[GroupID!=fold]
  y.test = y[GroupID==fold]
  train.rf = randomForest(x.train,y.train)
  predicted.y.matrix[GroupID==fold,1]=as.numeric(as.character(predict(train.rf,x.test)))
}

# get the important index for each wavelength (feature selection)
important.matrix=matrix(NA,243,5)
for (fold in 1:Fold){
  # fold = 1
  print(c(fold,Fold))
  x.train = x[GroupID!=fold,]
  x.test = x[GroupID==fold,]
  y.train = y[GroupID!=fold]
  y.test = y[GroupID==fold]
  train.rf = randomForest(x.train,y.train, importance = TRUE)
  print(train.rf[['importance']])
  print(importance(train.rf))
  important.matrix[,fold] = as.numeric(train.rf[['importance']][,'MeanDecreaseGini'])
}
write.table(important.matrix, file='important_index.csv', row.names=FALSE)

# support vector machine
for (fold in 1:Fold){
  # fold = 1
  print(c(fold,Fold))
  x.train = x[GroupID!=fold,]
  x.test = x[GroupID==fold,]
  y.train = y[GroupID!=fold]
  y.test = y[GroupID==fold]
  train.svm <- svm(x.train, y.train, probability = TRUE)
  pred_prob <- predict(train.svm, x.test, decision.values = TRUE, probability = TRUE)
  pred.prob.matrix=attr(pred_prob, "probabilities")
  labels = as.numeric(colnames(pred.prob.matrix))
  predicted.y.matrix[GroupID==fold,2] = labels[apply(pred.prob.matrix,1,which.max)]
}


# Discriminant function analysis
#Quadratic Discriminant Analysis (QDA) suppose the variance are differnt across classes
library(MASS)
for (fold in 1:Fold){
  # fold = 1
  print(c(fold,Fold))
  x.train = x[GroupID!=fold,]
  x.test = x[GroupID==fold,]
  y.train = y[GroupID!=fold]
  y.test = y[GroupID==fold]
  train.lda = MASS::lda(x.train, y.train)
  predicted.y.matrix[GroupID==fold,3] = as.numeric(as.character(predict(train.lda, x.test)$class))
  train.qda = MASS::qda(x.train, y.train)
  predicted.y.matrix[GroupID==fold,4] = as.numeric(as.character(predict(train.qda, x.test)$class))
}

# partial least squares-discriminant analysis
for (fold in 1:Fold){
  # fold = 1
  print(c(fold,Fold))
  x.train = x[GroupID!=fold,]
  x.test = x[GroupID==fold,]
  y.train = y[GroupID!=fold]
  y.test = y[GroupID==fold]
  plsda.train=plsda(x.train,y.train,ncomp=10)
  predicted.y.matrix[GroupID==fold,5] = as.numeric(as.character(predict(plsda.train, x.test)))
}

# multinomial logistic regression
for (fold in 1:Fold){
  # fold = 1
  print(c(fold,Fold))
  x.train = x[GroupID!=fold,]
  x.test = x[GroupID==fold,]
  y.train = y[GroupID!=fold]
  y.test = y[GroupID==fold]
  # MLR
  bwt=data.frame(x.train,y.train)
  multinom.mu <- multinom(y.train ~ ., bwt)
  predicted.y.matrix[GroupID==fold,6]=as.factor(as.character(predict(multinom.mu,x.test)))
}

# LASSO regression
for (fold in 1:Fold){
  print(c(fold,Fold))
  x.train = x[GroupID!=fold,]
  x.test = x[GroupID==fold,]
  y.train = y[GroupID!=fold]
  y.test = y[GroupID==fold]
  # LASSO for factors (classification)
  x.train.matrix=as.matrix(x.train)
  x.test.matrix=as.matrix(x.test)
  # glmnet expect the input as a matrix not a data frame
  # glmnet: fit a general linear model with Lasso or Elastic net Regularization
  # LASSO, default value is used.
  lasso.train=glmnet(x.train.matrix, y.train, family="multinomial") 
  #alpha=1 is the lasso penalty, and alpha=0 the ridge penalty.defauly alpha=1
  #family = binomial will perform binary classifiction
  predicted.y.matrix[GroupID == fold,7] = as.numeric(as.character(predict(lasso.train, newx = x.test.matrix,
                                                                          type="class", s=0.01)))
}

save.image("Results_7models.RData")
write.table(predicted.y.matrix,file="Results.csv",sep=",",quote=FALSE,col.name=FALSE,row.name=FALSE)
