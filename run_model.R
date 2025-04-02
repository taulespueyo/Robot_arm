
# run_model.R

# Load required libraries
library(randomForest)  # or your actual model package
library(readr)         # for reading CSV files
library(dplyr)         # optional, for data wrangling

# Path to model file (adjust name if needed)
model_path <- "models/robot_rf.rds"

# Load the trained model
if (!file.exists(model_path)) {
  stop("Model file not found: ", model_path)
}
model <- readRDS(model_path)

# Path to input data file (adjust file name!)
input_data_path <- "data/robot_df1.csv"

# Load input data
if (!file.exists(input_data_path)) {
  stop("Input data file not found: ", input_data_path)
}
input_data <- read_csv(input_data_path)

# Preview input data
print("Input data:")
print(head(input_data))

# Run prediction
prediction <- predict(model, input_data)

# Output predictions
print("Predictions:")
print(prediction)

# Optionally save to file
output_path <- "data/predictions.csv"
write_csv(data.frame(prediction), output_path)
cat("Predictions saved to:", output_path, "\n")