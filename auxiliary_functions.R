library(MASS) # LDA
library(randomForest)
library(glmnet) # Regresión logística

######################
###   GENERATORS   ###
######################

gen_mean_list <- function(p, min = 1, max = 100) {
  return(runif(p, min = min, max = max))
}

gen_means_list <- function(p, minimax_list) {
  means = list()
  for (i in 1:length(minimax_list)) {
    min <- minimax_list[[i]][1]
    max <- minimax_list[[i]][2]
    means[[i]] <- gen_mean_list(p, min, max)
  }
  return(means)
}

gen_cov_matrix <- function(p) {
  m <- matrix(runif(p^2, min = -1, max = 1), p, p)
  m <- make_symmetrical(m)
  m <- make_positive_definite(m)
  return(m)
}

gen_data <- function(n, mean, cov_matrix) {
  return(mvrnorm(n, mean, cov_matrix))
}

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

make_symmetrical <- function(matrix) {
  return((matrix + t(matrix)) / 2)
}

make_positive_definite <- function(matrix) {
  p <- length(matrix[1,])
  return(matrix %*% t(matrix) + 1e-6 * diag(p))
}

###############################
###   AUXILIARY FUNCTIONS   ###
###############################

get_prior <- function(n) {
  return(rep(1, n) / n)
}

combine_x_data <- function(data_list) {
  combined_data <- do.call(rbind, data_list)
  return(combined_data)
}

create_y_labels <- function(n_classes, n) {
  return(factor(rep(1:n_classes, rep(n, n_classes))))
}

#########################################
###   PREDICTIONS AND ERROR GETTERS   ###
#########################################

get_miscr_by_lda <- function(x_train, y_train, x_test, y_test, n_classes) {
  lda_model <- lda(x_train, y_train, prior = get_prior(n_classes), method = "mle")
  classes <- predict(lda_model, x_test)$class
  return(mean(y_test != classes))
}

get_miscr_by_lr <- function(x_train, y_train, x_test, y_test) {
  ml <- glmnet(x_train, as.factor(y_train), alpha = 1, family = "multinomial", lambda = 0)
  classes <- predict(ml, x_test, s = 0.0001, type = "class")
  return(mean(y_test != classes))
}

get_miscr_by_lasso <- function(x_train, y_train, x_test, y_test) {
  cv.lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "multinomial")
  l <- cv.lasso$lambda.min
  model <- glmnet(x_train, y_train, alpha = 1, family = "multinomial", lambda = l, standardized = FALSE)
  classes <- predict(model, x_test, s = 0.0001, type = "class")
  return(mean(y_test != classes))
}

get_miscr_by_rf  <- function(x_train, y_train, x_test, y_test) {
  rf <- randomForest(x_train, y_train, importance = TRUE)
  classes <- predict(rf, x_test, type = "class")
  return(mean(y_test != classes))
}
