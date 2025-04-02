
# packages.R

# Install required packages (if not installed)
packages <- c("caret", "randomForest", "ggplot2", "readxl", "rpart", "dslabs", "doparallel")

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

invisible(sapply(packages, install_if_missing))