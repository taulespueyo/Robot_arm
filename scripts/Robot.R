
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

robot <- list.files(path = "C:/Users/uic53995/OneDrive - Continental AG/Dokumente/Privat/Robot/", pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ read_csv(., col_names = FALSE)) 
class(robot)

# Read header file and header sheet

header <- (read_excel("C:/Users/uic53995/OneDrive - Continental AG/Dokumente/Privat/Robot/01_ur5testresult_header.xlsx", sheet = "header")) %>% 
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
acc_M1 <- postResample(pred = unname(predict(robot_rf, test_robot)), obs = test_robot$pos_dev_6) # Acc about 0.6, overfitted with 500 tres and 25 mtry



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
stopCluster(cl)


# Model 2






