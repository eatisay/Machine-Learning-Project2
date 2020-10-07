#Reading images and labels
imgs<-read.csv("hw02_images.csv",header = FALSE)
lbls<-read.csv("hw02_labels.csv",header = FALSE)
#Reading weight values
W<-read.csv("initial_W.csv",header = FALSE)
W0<-read.csv("initial_w0.csv",header = FALSE)
#Partitioning the image data
trainSet<-imgs[c(1:(nrow(imgs)/2)),]
testSet<-imgs[c((nrow(imgs)/2)+1):nrow(imgs),]
dataSet<-cbind(imgs,lbls)
#Partitioning the label data
yTrain<-lbls[c(1:(nrow(lbls)/2)),]
yTest<-lbls[c((nrow(lbls)/2)+1):nrow(lbls),]
#To have a w0 as row
w0<-t(as.matrix(W0))
#w0 is replicated 500 times
w0<-do.call("rbind", replicate(500, w0, simplify = FALSE))
#To refresh the w0 as it is needed in the loops
w0<-w0500
#Sigmoid Function
sigmoidFunc <- function(X, w, w0) {
  return (1 / (1 + exp(-(as.matrix(X) %*% as.matrix(w) + w0))))
}
#Constants for the algorithm
eta<- 0.0001
epsilon<-1e-3
maxIteration<- 500
#Calculation of gradientW
gradientW <- function(X, Ytruth, Ypredicted) {
  tmp<-(Ytruth - Ypredicted)*Ypredicted*(1 - Ypredicted)
  XT<-t(X)
  return(-(as.matrix(XT)%*%as.matrix(tmp)))
}
#Calculation of gradientw0
gradientw0 <- function(Ytruth, Ypredicted) {
  return (-colSums((Ytruth - Ypredicted)*Ypredicted*(1 - Ypredicted)))
}
#Label values are changed in order to have matrix representation in binary form as one hot encoding
yTrainOptimazed<-data.frame()
yTestOptimazed<-data.frame()
a<-1
while(a<=length(yTest)){
  if(yTest[a]==1) yTestOptimazed<-rbind(yTestOptimazed,c(1,0,0,0,0))
  else if(yTest[a]==2)yTestOptimazed<-rbind(yTestOptimazed,c(0,1,0,0,0))
  else if(yTest[a]==3)yTestOptimazed<-rbind(yTestOptimazed,c(0,0,1,0,0))
  else if(yTest[a]==4)yTestOptimazed<-rbind(yTestOptimazed,c(0,0,0,1,0))
  else if(yTest[a]==5) yTestOptimazed<-rbind(yTestOptimazed,c(0,0,0,0,1))
  print(a)
  a<-a+1
}
a<-1
while(a<=length(yTrain)){
  if(yTrain[a]==1) yTrainOptimazed<-rbind(yTrainOptimazed,c(1,0,0,0,0))
  if(yTrain[a]==2)yTrainOptimazed<-rbind(yTrainOptimazed,c(0,1,0,0,0))
  if(yTrain[a]==3)yTrainOptimazed<-rbind(yTrainOptimazed,c(0,0,1,0,0))
  if(yTrain[a]==4)yTrainOptimazed<-rbind(yTrainOptimazed,c(0,0,0,1,0))
  if(yTrain[a]==5) yTrainOptimazed<-rbind(yTrainOptimazed,c(0,0,0,0,1))
  a<-a+1
}

#Iterative Algorithm for the training data set
iteration <- 0
objectiveValuesTraining <- c()
while (iteration<maxIteration) {
  #Y predicted calculated from the sigmoid function
  Ypredicted <- sigmoidFunc(trainSet, W, w0)
  #Onjective value is the deviation from the real value, matrix representation of training label
  objectiveValuesTraining <- c(objectiveValuesTraining, 0.5*sum(colSums((yTrainOptimazed - Ypredicted)^2)))
  #W and w0 are updated to have better prediction in the following iteration
  W <- W - eta * gradientW(trainSet, yTrainOptimazed, Ypredicted)
  w0 <- w0 - eta * gradientw0(yTrainOptimazed, Ypredicted)
  iteration <- iteration + 1
  print(iteration)
}
#Figure of error versus iteration
plot(1:iteration, objectiveValuesTraining,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")
#Y predicted calculated from the max between the columns for evry observation. One column always converges to 1
ypredicted <- apply(Ypredicted, MARGIN = 1, FUN = which.max)
#Confusion matrix of the training data and predicted data for the them
confusionMatrixTraining <- table(ypredicted, yTrain)
print(confusionMatrixTraining)

#For test data, the last versions of the W and w0 from the iteration of the train data is used.
#Again, predicted values are taken from the the sigmoid function, which inputs those updated weights
YpredictedTest <- sigmoidFunc(testSet, W, w0)
ypredictedTest <- apply(YpredictedTest, MARGIN = 1, FUN = which.max)
confusionMatrixTest <- table(ypredictedTest, yTest)
print(confusionMatrixTest)

