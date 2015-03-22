rm(list=ls(all=TRUE))

library(caret)
library(randomForest)

training = read.csv("C:\\classes\\Coursera\\Data_Science_Specialization\\Practical_Machine_Learning\\project\\pml-training.csv")
testing = read.csv("C:\\classes\\Coursera\\Data_Science_Specialization\\Practical_Machine_Learning\\project\\pml-testing.csv")

#summary(training)

set.seed(32323)

# set a logical vector to indicate which attributes we want to get rid of
attributes_to_discard = logical(ncol(training))
attributes_to_discard[c(1, 3:7)] = TRUE
# get rid of attributes that begins with "total_"
attributes_to_discard = attributes_to_discard | grepl("^total_", names(training)) # make sure it begins with "total_"

# removing zero covariates
nsv = nearZeroVar(training, saveMetrics=TRUE)
attributes_to_discard = attributes_to_discard | nsv$zeroVar| nsv$nzv
#training_subset = training[!(nsv$zeroVar|nsv$nzv)]


# remove the attributes with mostly NA (> 30%)
# we'll start with 
num_obs = nrow(training)
# start from 7th column to the 2nd from last (classe)
for (ii in 8:ncol(training)-1) {
    if (sum(is.na(training[,ii]))/num_obs >= 0.3) {
        attributes_to_discard[ii] = TRUE
    }
}
training_subset=training[,!attributes_to_discard]


# since there's a wide range of variance between individuals for the attributes
# I assume this predictor should be calibrated individually. Therefore, the
# taining and prediction will be done individually
unique_users = unique(training_subset$user_name)
num_attributes = ncol(training_subset)
modelFit_list = list()
var_importance = list()
cv_result = list()
for (ii in 1:length(unique_users)) {
    # try Random Forest
    user_training_data = training_subset[training_subset$user_name == unique_users[ii],]
    modelFit = train(classe ~., data=user_training_data[,-1], method = "rf", prox = TRUE)
    modelFit
    
    modelFit_list[[ii]] = modelFit  # store one model per user name
    var_importance[[ii]] = varImp(modelFit, useModel = TRUE)
    cv_result[[ii]] = rfcv(user_training_data[,-1], user_training_data$classe, cv.fold=3)
    
    with(cv_result[[ii]], plot(n.var, error.cv, log="x", type="o", lwd=2, main=unique_users[ii]))
    print(unique_users[ii])
    print(cv_result[[ii]]$error.cv)
    
}

# for (ii in 1:length(unique_users)) {
#     var_importance[[ii]] = varImp(modelFit_list[[ii]], useModel = TRUE)
#     user_training_data = training_subset[training_subset$user_name == unique_users[ii],]
#     cv_result[[ii]] = rfcv(user_training_data[,-1], user_training_data$classe, cv.fold=10)
#     with(cv_result[[ii]], plot(n.var, error.cv, log="x", type="o", lwd=2, main=unique_users[ii]))
#     print(unique_users[ii])
#     print(cv_result[[ii]]$error.cv)
# }


# check the cases that it was predicted incorrectly with the training set
for (ii in 1:length(unique_users)) {
    user_training_data = training_subset[training_subset$user_name == unique_users[ii],]
    pred = predict(modelFit_list[[ii]], user_training_data[,-1])
    predRight = pred == user_training_data$classe
    print(unique_users[ii])
    print(table(pred, predRight))
}


# # plot the centers of the classes by magnet_dumbbell_z and yaw_belt
# userP = classCenter(user_training_data[c(37,4)], user_training_data$classe, modelFit_list[[1]]$finalModel$prox)
# userP = as.data.frame(userP); userP$classe = rownames(userP)
# p = qplot(magnet_dumbbell_z, yaw_belt, col=classe, data=user_training_data)
# p + geom_point(aes(x=magnet_dumbbell_z, y=yaw_belt, col=classe), size=5, shape=4, data=userP)

# get rid of the attributes that were not used in training
testing_subset=testing[,!attributes_to_discard]

test_pred = vector()
# prediction
for (ii in 1:nrow(testing_subset)) {
    user_num = which(unique_users == testing_subset$user_name[ii])
    test_pred = c(test_pred, as.character(predict(modelFit_list[[user_num]], testing_subset[ii,-1])))
    print(unique_users[user_num])
    print(as.character(predict(modelFit_list[[user_num]], testing_subset[ii,-1])))
    
}

save.image("C:\\classes\\Coursera\\Data_Science_Specialization\\Practical_Machine_Learning\\project\\all_user_models.RData")

################
# submission
################
file_dir = "C:\\classes\\Coursera\\Data_Science_Specialization\\Practical_Machine_Learning\\project\\my_test_set_predictions\\"
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=paste(file_dir, filename, sep=""),quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(test_pred)


