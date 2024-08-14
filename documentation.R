#library(MASS) # LDA
#library(randomForest)
#library(glmnet) # Regresión logística

######################
###   GENERATORS   ###
######################

# Generates a list of random means.
# 
# Args:
#   p: Number of means to generate.
#   min: Minimum value for the generated means (default is 1).
#   max: Maximum value for the generated means (default is 100).
# 
# Returns:
#   A vector of size p with random values between min and max.
gen_mean_list <- function(p, min = 1, max = 100) {
  return(runif(p, min = min, max = max))
}

# Generates a list of lists containing random means, each with different bounds.
# 
# Args:
#   p: Number of means in each list.
#   minimax_list: A list of pairs (min, max) defining the bounds for each set of means.
# 
# Returns:
#   A list where each element is a vector of random means generated according to the specified bounds.
gen_means_list <- function(p, minimax_list) {
  means = list()
  for (i in 1:length(minimax_list)) {
    min <- minimax_list[[i]][1]
    max <- minimax_list[[i]][2]
    means[[i]] <- gen_mean_list(p, min, max)
  }
  return(means)
}

# Generates a random positive definite covariance matrix.
# 
# Args:
#   p: Dimension of the matrix (p x p).
# 
# Returns:
#   A covariance matrix of size p x p that is symmetric and positive definite.
gen_cov_matrix <- function(p) {
  m <- matrix(runif(p^2, min = -1, max = 1), p, p)
  m <- make_symmetrical(m)
  m <- make_positive_definite(m)
  return(m)
}

# Generates a multivariate dataset from a mean vector and a covariance matrix.
# 
# Args:
#   n: Number of observations to generate.
#   mean: A vector of means.
#   cov_matrix: A covariance matrix.
# 
# Returns:
#   A data matrix of size n x p, where p is the dimension of the mean vector.
gen_data <- function(n, mean, cov_matrix) {
  return(mvrnorm(n, mean, cov_matrix))
}

# Generates a list of multivariate datasets from lists of means and a covariance matrix.
# 
# Args:
#   n: Number of observations for each dataset.
#   means: A list of mean vectors.
#   cov_matrix: A covariance matrix.
# 
# Returns:
#   A list where each element is a dataset generated using the corresponding mean vector and the covariance matrix.
gen_data_list <- function(n, means, cov_matrix) {
  data_list = list()
  for (c in 1:length(means)) {
    data_list[[c]] <- mvrnorm(n, means[[c]], cov_matrix)
  }
  return(data_list)
}

##########################################
###   COV MATRIX AUXILIARY FUNCTIONS   ###
##########################################

# Makes a matrix symmetric by averaging it with its transpose.
# 
# Args:
#   matrix: A square matrix.
# 
# Returns:
#   A symmetric matrix obtained by averaging the input matrix with its transpose.
make_symmetrical <- function(matrix) {
  return((matrix + t(matrix)) / 2)
}

# Ensures that a matrix is positive definite by adding a small value to its diagonal.
# 
# Args:
#   matrix: A square matrix.
# 
# Returns:
#   A positive definite matrix created by multiplying the matrix by its transpose and adding a small value to its diagonal.
make_positive_definite <- function(matrix) {
  p <- length(matrix[1,])
  return(matrix %*% t(matrix) + 1e-6 * diag(p))
}

###############################
###   AUXILIARY FUNCTIONS   ###
###############################

# Generates a uniform prior distribution for a given number of classes.
# 
# Args:
#   n: Number of classes.
# 
# Returns:
#   A vector of size n where each element is 1/n, representing a uniform prior distribution.
get_prior <- function(n) {
  return(rep(1, n) / n)
}

# Combines a list of datasets into a single dataset by row-binding.
# 
# Args:
#   data_list: A list of data matrices to combine.
# 
# Returns:
#   A single data matrix obtained by row-binding all matrices in the data_list.
combine_x_data <- function(data_list) {
  combined_data <- do.call(rbind, data_list)
  return(combined_data)
}

# Creates factor labels for a multi-class classification problem.
# 
# Args:
#   classes: Number of classes.
#   n: Number of observations per class.
# 
# Returns:
#   A factor vector with labels 1 to classes, each repeated n times.
create_y_labels <- function(classes, n) {
  return(factor(rep(1:classes, rep(n, classes))))
}

#########################################
###   PREDICTIONS AND ERROR GETTERS   ###
#########################################

# Computes the misclassification rate using Linear Discriminant Analysis (LDA).
# 
# Args:
#   x_train: Training data matrix.
#   y_train: Training labels.
#   x_test: Test data matrix.
#   y_test: Test labels.
#   classes: Number of classes.
# 
# Returns:
#   The misclassification rate on the test data.
get_miscr_by_lda <- function(x_train, y_train, x_test, y_test, classes) {
  lda_model <- lda(x_train, y_train, prior = get_prior(classes), method = "mle")
  classes <- predict(lda_model, x_test)$class
  return(mean(y_test != classes))
}

# Computes the misclassification rate using Logistic Regression.
# 
# Args:
#   x_train: Training data matrix.
#   y_train: Training labels.
#   x_test: Test data matrix.
#   y_test: Test labels.
# 
# Returns:
#   The misclassification rate on the test data.
get_miscr_by_lr <- function(x_train, y_train, x_test, y_test) {
  ml <- glmnet(x_train, as.factor(y_train), alpha = 1, family = "multinomial", lambda = 0)
  classes <- predict(ml, x_test, s = 0.0001, type = "class")
  return(mean(y_test != classes))
}

# Computes the misclassification rate using LASSO (Least Absolute Shrinkage and Selection Operator).
# 
# Args:
#   x_train: Training data matrix.
#   y_train: Training labels.
#   x_test: Test data matrix.
#   y_test: Test labels.
# 
# Returns:
#   The misclassification rate on the test data.
get_miscr_by_lasso <- function(x_train, y_train, x_test, y_test) {
  cv.lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "multinomial")
  l <- cv.lasso$lambda.min
  model <- glmnet(x_train, y_train, alpha = 1, family = "multinomial", lambda = l, standardized = FALSE)
  classes <- predict(model, x_test, s = 0.0001, type = "class")
  return(mean(y_test != classes))
}

# Computes the misclassification rate using Random Forest.
# 
# Args:
#   x_train: Training data matrix.
#   y_train: Training labels.
#   x_test: Test data matrix.
#   y_test: Test labels.
# 
# Returns:
#   The misclassification rate on the test data.
get_miscr_by_rf  <- function(x_train, y_train, x_test, y_test) {
  rf <- randomForest(x_train, y_train, importance = TRUE)
  classes <- predict(rf, x_test, type = "class")
  return(mean(y_test != classes))
}
