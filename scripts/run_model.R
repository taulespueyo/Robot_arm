
# run_model.R

# Load required libraries
library(randomForest)  # or your actual model package
library(readr)         # for reading CSV files
library(dplyr)         # optional, for data wrangling
library(tidyverse)
library(readxl)

# Path to model file (adjust name if needed)
model_path <- "models/robot_rf.rds"

# Load the trained model
if (!file.exists(model_path)) {
  stop("Model file not found: ", model_path)
}
model <- readRDS(model_path)

## Path to input data file (adjust file name!)
#input_data_path <- "data/robot_df1.csv"
#
## Load input data
#if (!file.exists(input_data_path)) {
#  stop("Input data file not found: ", input_data_path)
#}
#input_data <- read_csv(input_data_path)

# Read data set

robot <- list.files(path = "data/", pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ read_csv(., col_names = FALSE)) 
class(robot)

# Read header file and header sheet

header <- (read_excel("data/01_ur5testresult_header.xlsx", sheet = "header")) %>% 
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

# Define output y and create data partition

for (i in 1:6){
  column_name <- paste0("dev_", i)
  robot_df_dev <- robot_df1 %>% mutate(!!sym(paste0("pos_dev_",i)) := .data[[paste0("ROBOT_ACTUAL_JOINT_POSITIONS..J",i,".")]] - .data[[paste0("ROBOT_TARGET_JOINT_POSITIONS..J",i,".")]],
                                       !!sym(paste0("vel_dev_",i)) := .data[[paste0("ROBOT_ACTUAL_JOINT_POSITIONS..J",i,".")]] - .data[[paste0("ROBOT_TARGET_JOINT_POSITIONS..J",i,".")]],
                                       !!sym(paste0("tor_dev_",i)) := .data[[paste0("ROBOT_ACTUAL_JOINT_POSITIONS..J",i,".")]] - .data[[paste0("ROBOT_TARGET_JOINT_POSITIONS..J",i,".")]],
                                       !!sym(paste0("curr_dev_",i)) := .data[[paste0("ROBOT_ACTUAL_JOINT_POSITIONS..J",i,".")]] - .data[[paste0("ROBOT_TARGET_JOINT_POSITIONS..J",i,".")]])
}





# Preview input data
print("Input data:")
print(head(robot_df1))

# Run prediction
prediction <- predict(model, robot_df1)

# Output predictions
print("Predictions:")
print(prediction)

# Optionally save to file
output_path <- "data/predictions.csv"
write_csv(data.frame(prediction), output_path)
cat("Predictions saved to:", output_path, "\n")
