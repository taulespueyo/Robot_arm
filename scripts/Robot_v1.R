
library(tidyverse)
library(readxl)
library(caret)
library(randomForest)
# load package for decision tree
library(rpart)

# load the dslabs package
library(dslabs)

# Install and load parallel and doParallel packages
library(doParallel)


# Read data set

robot <- list.files(path = "~/datasets/home-dataset/Robot/", pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ read_csv(., col_names = FALSE)) 
class(robot)

# Read header file and header sheet

header <- (read_excel("~/datasets/home-dataset/Robot/01_ur5testresult_header.xlsx", sheet = "header")) %>% 
  mutate(across(everything(), as.character))
class(header)

# Allocate the header file to the names of the columns in the robot data and convert to data frame

colnames(robot) <- names(header)
robot_df <- robot %>% as.data.frame()

# Remove breaks and convert to numeric values

robot_df1 <- robot_df %>% 
  mutate(across(where(~ !is.numeric(.)), ~ gsub("\\(|\\)|\\[|\\]", "", .))) %>% 
  lapply(function(x) as.numeric(as.character(x)))

robot_df1 <- robot_df1 %>% as.data.frame()

str(robot_df1)

# Save data frame as csv file

write.csv(robot_df1, "robot_df1.csv", row.names = FALSE)

# Read csv file

robot_df1 <- read.csv("robot_df1.csv")

# Define output y and create data partition

for (i in 1:6){
  column_name <- paste0("dev_", i)
  robot_df1 <- robot_df1 %>% mutate(!!sym(paste0("pos_dev_",i)) := .data[[paste0("ROBOT_ACTUAL_JOINT_POSITIONS..J",i,".")]] - .data[[paste0("ROBOT_TARGET_JOINT_POSITIONS..J",i,".")]],
                                    !!sym(paste0("vel_dev_",i)) := .data[[paste0("ROBOT_ACTUAL_JOINT_POSITIONS..J",i,".")]] - .data[[paste0("ROBOT_TARGET_JOINT_POSITIONS..J",i,".")]],
                                    !!sym(paste0("tor_dev_",i)) := .data[[paste0("ROBOT_ACTUAL_JOINT_POSITIONS..J",i,".")]] - .data[[paste0("ROBOT_TARGET_JOINT_POSITIONS..J",i,".")]],
                                    !!sym(paste0("curr_dev_",i)) := .data[[paste0("ROBOT_ACTUAL_JOINT_POSITIONS..J",i,".")]] - .data[[paste0("ROBOT_TARGET_JOINT_POSITIONS..J",i,".")]])
}




#_____ MODEL 1:a deviation model that predicts deviation based only on temperature, force, velocities and tool position______
#____________________________________________________________________________________________________________________________

#Define input and output for predicting the accuracy of Joint 6, based only on temperature, force, velocities and tool position
# Split data into train and test data
# No PCA required (correlated inputs are droped manually)

y_1 <- robot_df1["pos_dev_6"]
col_toSelect <- grep("TEMP|FORCE|VELOCITIES|\\.x\\.|\\.y\\.|\\.z\\.", names(robot_df1), value = TRUE)
x_1 <- robot_df1[col_toSelect]
robot_M1 <- data.frame(x_1, y_1)

test_index <- createDataPartition(robot_M1$pos_dev_6, times = 1, p = 0.8,  list = FALSE)
test_robot <- robot_M1[-test_index, ]
train_robot <- robot_M1[test_index, ] 

# Detect number of cores in your machine and use all but one cores
cores <- detectCores() - 1  # Use one less than the total number of cores
cl <- makeCluster(cores)
registerDoParallel(cl)


# Approach A
# Train the random forest model and calculate accuracy on test set

robot_rf <- randomForest(train_robot$pos_dev_6 ~ ., data=train_robot, ntree = 150, mtry = 25)
rmse_M1 <- sqrt(mean((unname(predict(robot_rf, test_robot)) - test_robot$pos_dev_6)^2))
acc_M1 <- postResample(pred = unname(predict(robot_rf, test_robot)), obs = test_robot$pos_dev_6) # Acc about 0.6, overfitted with 500 tres and 25 mtry; Acc 0.87 with 150 trees and 25 mtry



# Approach B
# Tune random forest model and calculate accuracy on test set

# rf_tune_M1 <- tuneRF(
#   train_robot[, -1],    # Independent variables (excluding target variable 'mpg')
#   train_robot$pos_dev_6,      # Target variable
#   stepFactor = 1.5,    # Step factor for increasing 'mtry'
#   improve = 0.01,      # How much the improvement should be to stop tuning
#   ntreeTry = 100,      # Number of trees to start with
#   trace = TRUE         # Print progress of the tuning process
# )
# # Save results of Model 1
# save(rf_tune,file = "rf_tune_M1.RData")
# 
# # Make predictions on the test set with the best parameters
# best_rf__M1 <- randomForest(
#   pos_dev_6 ~ ., data = train_robot, 
#   mtry = rf_tune_M1[1, 1], 
#   ntree = rf_tune_M1[1, 2]
# )
# acc_M1 <- postResample(pred = unname(predict(robot_rf, test_robot)), obs = test_robot$pos_dev_6)



# Approach C
# Define control with progress tracking, train the random forest model and calculate accuracy on test set
train_control <- trainControl(method="cv", number = 5, verboseIter = TRUE)  # Enable progress updates
grid <- data.frame(mtry = c(1, 5, 10, 25, 50, 100))
rf_model <- train(robot_M1$pos_dev_6 ~ ., data = robot_M1, method = "rf", ntree = 150, tuneGrid = grid, trControl = train_control)
acc_M1 <- postResample(pred = unname(predict(rf_model, test_robot)), obs = test_robot$pos_dev_6)


# Stop the parallel cluster after training
# stopCluster(cl)


#_____ MODEL 2:a deviation model that predicts deviation using all deviations, as well as temperature, force, velocities and tool position______
#_______________________________________________________________________________________________________________________________________________

#Define input and output for predicting the accuracy of Joint 6, based only on temperature, force, velocities and tool position
# Split data into train and test data
# PCA required to drop highly correlated inputs

# Detect number of cores in your machine and use all but one cores
cores <- detectCores() - 1  # Use one less than the total number of cores
cl <- makeCluster(cores)
registerDoParallel(cl)
numCores <- cores

# Read data set and split into test and training set
robot_df2 <- robot_df1

y_2 <- robot_df2["pos_dev_6"]
test_index <- createDataPartition(robot_df2$pos_dev_6, times = 1, p = 0.8,  list = FALSE)
test_robot2 <- robot_df2[-test_index, ]
train_robot2 <- robot_df2[test_index, ] 

# Remove low-variance variables
low_var_indices <- nearZeroVar(robot_df2[, !names(robot_df2) %in% "pos_dev_6"])
robot_filtered <- robot_df2[, -low_var_indices]

# Remove highly correlated variables (threshold = 0.9)
correlation_matrix <- cor(robot_filtered)
high_cor_indices <- findCorrelation(correlation_matrix, cutoff = 0.9)
robot_filtered_corr <- robot_filtered[, -high_cor_indices]

# Calculate PCA using centered, scaled and filtered trainings data and perform this PCA on both the training set and the test set
train_robot_scaled <- scale(robot_filtered_corr[test_index, ])
test_robot_scaled <- scale(robot_filtered_corr[-test_index, ])
robot_PCA <- prcomp(train_robot_scaled, center = TRUE, scale. = TRUE)
#robot_PCA <- as.data.frame(robot_PCA)

train_pca2 <- predict(robot_PCA, newdata = train_robot_scaled )
postResample(pred = train_pca2, obs = robot_PCA$x)  # test whether the PCA conversion is working on the scaled trainings set
test_pca2 <- predict(robot_PCA, newdata = test_robot_scaled)

robot_PCA$x

# Random forest using PCs
ntree <- 150
robot_rf_PCA <- foreach(ntree = rep(ntree / numCores, numCores), .combine = combine, .packages = 'randomForest') %dopar% {
  randomForest(robot_PCA$x[, 1:ncol(robot_PCA$x)], train_robot2["pos_dev_6"][, 1], ntree = ntree, mtry = 10)
}
#reversed_robot_rf <- as.matrix(robot_rf_PCA) %*% t(robot_PCA$rotation[, 1:ncol(robot_PCA$rotation)])
acc_test <- postResample(pred = predict(robot_rf_PCA, train_pca2), obs = train_robot2$pos_dev_6)
acc_M2 <- postResample(pred = predict(robot_rf_PCA, test_pca2), obs = test_robot2$pos_dev_6)  # Accuracy 0.818 with 150 tres and 25 mtry; Accuracy 0.795 with 300 tres and 20 mtry (both overfitted to training set); Accuracy 0.769 with 150 trees and 10 mtry (overfitted to training set)

