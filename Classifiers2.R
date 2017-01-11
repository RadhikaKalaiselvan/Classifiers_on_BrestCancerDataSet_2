#install.packages('randomForest')
#install.packages('class')
#install.packages('ipred')
#install.packages('rpart')
#install.packages('adabag')
#install.packages('ada')
#install.packages('ROCR')
#library(randomForest)
#library(rpart)
#library(ipred)
#library(adabag)
#library(class)
#library(ada)
#library(stats)
#library(ROCR)
require(randomForest)
require(rpart)
require(ipred)
require(adabag)
require(class)
require(ada)
require(stats)
require(ROCR)

#Load the data into a data frame
breastCancerDataWithNA<-read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=FALSE,sep=",")
#Omit the data with NA values
breastCancerData<-na.omit(breastCancerDataWithNA)

#Convert the class( M malignant and B Benign) into malignant as 1 and Benign as 0 
Malign=ifelse(breastCancerData$V2=="M",1,0)
#Add column Malign to the end of the dataframe
breastCancerData=data.frame(breastCancerData,Malign)
#Remove column V2 (old string predicted values) from the dataframe 
breastCancerData=breastCancerData[-2]

#Rename the attribute names in the data set 
names(breastCancerData)<-c("ID","RV1","RV2","RV3","RV4","RV5","RV6","RV7","RV8","RV9","RV10","RV11","RV12",
                           "RV13","RV14","RV15","RV16","RV17","RV18","RV19",
                           "RV20","RV21","RV22","RV23","RV24","RV25","RV26","RV27","RV28","RV29","RV30","Diagnosis")

print(paste("No of instances =",nrow(breastCancerData)))
print(paste("No of attributes =",ncol(breastCancerData)-1))
ClassIndex <- length(breastCancerData)
rows=nrow(breastCancerData)

#No of cross validation folds
kFolds= 10
print(paste("No of cross validation folds =", kFolds))

#create k-folds
id <- sample(1:kFolds,nrow(breastCancerData),replace=TRUE)
list <- 1:kFolds

#Logistic regression
sumAccuracy = 0
accuracy=0
predVector <- c()
testVector <- c()
print("Logistic Regression")
for (i in 1:kFolds)
{
  trainingData <- subset(breastCancerData, id %in% list[-i])
  testData <- subset(breastCancerData, id %in% c(i))
  class<-trainingData[,ClassIndex]
  set.seed(415)
  formula <- as.formula(paste("as.factor(",colnames(breastCancerData)[length(breastCancerData)],") ~","." ))
  lr <- glm(formula,data=trainingData,family=binomial(),model = TRUE, maxit = 200)
  predictedClass<-predict(lr,testData)
  predictedClass <- ifelse(predictedClass > 0.5,1,0)
  predVector <- c(predVector, predictedClass)
  testVector <- c(testVector, testData$Diagnosis)
  misClassifiedError <- mean(predictedClass != testData$Diagnosis)
  accuracy<-1-misClassifiedError
  sumAccuracy <- sumAccuracy + (accuracy*100)
}
acc <- sumAccuracy/kFolds
pred <- prediction(predictions = predVector, labels = testVector)
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
print(paste("Accuracy = ", acc))
print(paste("AUC = ", auc))


#K Nearest Neighbour
print("K Nearest Neighbour")
sumAccuracy = 0
accuracy=0
predVector <- c()
testVector <- c()
acc=0
#Scale the data to increase accuracy
maxs =	apply(breastCancerData,	MARGIN	=	2,	max)
mins =	apply(breastCancerData,	MARGIN	=	2,	min)
scaled =	 as.data.frame(scale(breastCancerData,	center	=	mins,	scale	=	maxs	- mins))

for (i in 1:kFolds)
{
trainingData <- subset(scaled, id %in% list[-i])
testData <- subset(scaled, id %in% c(i))
class<-trainingData[,ClassIndex]
set.seed(415)
model.knn <- knn(trainingData[,1:(ClassIndex-1)], testData[,1:(ClassIndex-1)],trainingData[,ClassIndex],k = 10, l = 0, prob = FALSE, use.all = TRUE)
tb_name <- table(Predictions = model.knn, Actual = testData[,ClassIndex])
predVector <- c(predVector, model.knn)
testVector <- c(testVector, testData$Diagnosis)
accuracy <- (sum(diag(tb_name)) / sum(tb_name))*100.0
sumAccuracy <- sumAccuracy + accuracy
}
acc <- sumAccuracy/kFolds
pred <- prediction(predictions = predVector, labels = testVector)
print(paste("Accuracy = ", acc))
auc.tmp <- performance(pred,"auc");
auc <- as.numeric(auc.tmp@y.values)
print(paste("AUC = ", auc))

#Bagging
print("Bagging")
sumAccuracy = 0
accuracy=0
acc=0
predVector <- c()
testVector <- c()
for (i in 1:kFolds)
{
trainingData <- subset(breastCancerData, id %in% list[-i])
testData <- subset(breastCancerData, id %in% c(i))
formula <- as.formula(paste("as.factor(",colnames(breastCancerData)[length(breastCancerData)],") ~","." ))
bag <- ipred::bagging(formula, data=trainingData, boos = TRUE,mfinal=10,nbagg = 60,
                      control = rpart.control(cp = 0.01, minsplit = 30, maxdepth = 20 )) 
predictedClass<-predict(bag,testData)
predVector <- c(predVector, predictedClass)
testVector <- c(testVector, testData$Diagnosis)
accuracy<- (sum(predictedClass==testData[,ClassIndex]))/length(testData[,ClassIndex])*100.0
sumAccuracy <- sumAccuracy + accuracy
}
acc <- sumAccuracy/kFolds
pred <- prediction(predictions = predVector, labels = testVector)
auc.tmp <- performance(pred,"auc");
auc <- as.numeric(auc.tmp@y.values)
print(paste("Accuracy = ", acc))
print(paste("AUC = ", auc))


#Random Forest
print("Random Forest")
sumAccuracy = 0
accuracy=0
predVector <- c()
testVector <- c()
for (i in 1:kFolds)
{
  trainingData <- subset(breastCancerData, id %in% list[-i])
  testData <- subset(breastCancerData, id %in% c(i))
  class<-trainingData[,ClassIndex]
  set.seed(415)
  rf <- randomForest(as.factor(class) ~ .,data=trainingData,importance=TRUE,ntree=200, mtry = 10 )
  predictedClass<-predict(rf,testData)
  predVector <- c(predVector, predictedClass)
  testVector <- c(testVector, testData$Diagnosis)
  accuracy<- (sum(predictedClass==testData[,ClassIndex]))/length(testData[,ClassIndex])*100.0
  sumAccuracy <- sumAccuracy + accuracy
}
acc <- sumAccuracy/kFolds
pred <- prediction(predictions = predVector, labels = testVector)
auc.tmp <- performance(pred,"auc");
auc <- as.numeric(auc.tmp@y.values)
print(paste("Accuracy = ", acc))
print(paste("AUC = ", auc))

#Boosting
print("Boosting")
predVector <- c()
testVector <- c()
sumAccuracy=0
accuracy=0
for (i in 1:kFolds)
{
  trainingData <- subset(breastCancerData, id %in% list[-i])
  testData <- subset(breastCancerData, id %in% c(i))
  n <- names(trainingData)
  fmla <- as.formula(paste("Diagnosis ~", paste(n[!n %in% "Diagnosis"], collapse = " + ")))
  adaboost <- ada(fmla, data = trainingData, iter=30, nu=1,delta=0, type="discrete")
  predictedClass<-predict(adaboost,testData)
  predVector <- c(predVector, predictedClass)
  testVector <- c(testVector, testData$Diagnosis)
  accuracy<- (sum(predictedClass==testData[,ClassIndex]))/length(testData[,ClassIndex])*100.0
  sumAccuracy <- sumAccuracy + accuracy
}
acc <- sumAccuracy/kFolds
print(paste("Accuracy = ", acc))
pred <- prediction(predictions = predVector, labels = testVector)
auc.tmp <- performance(pred,"auc");
auc <- as.numeric(auc.tmp@y.values)
print(paste("AUC = ", auc))


