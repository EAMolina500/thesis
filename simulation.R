rm(list = ls())

source("~/thesis/settings.R")
source("~/thesis/auxiliary_functions.R")

#classes_list <- classes_20[1:3]
#classes_list <- classes_20[1:5]
classes_list <- classes_20[1:7]

### lists below get errors
#classes_list <- classes_20[1:15]
#classes_list <- classes_20

run_simulation <- function(simulations = 1) {
  n_features <- 200
  n_train_samples <- 15
  n_test_samples <- 200
  n_models <- 4
  n_classes <- length(classes_list)
  
  means <- gen_means_list(n_features, classes_list)
  sigma <- gen_cov_matrix(n_features)
  misc_rate <- matrix(0, simulations, n_models)
  set.seed(2)
  
  for (sim in 1:simulations) {
    # Generate training data
    train_data <- gen_data_list(n_train_samples, means, sigma)
    x_train <- combine_x_data(train_data)
    y_train <- create_y_labels(n_classes, n_train_samples)
    
    # Generate test data
    test_data <- gen_data_list(n_test_samples, means, sigma)
    x_test <- combine_x_data(test_data)
    y_test <- create_y_labels(n_classes, n_test_samples)
    
    # LDA (Análisis Discriminante Lineal)
    misc_rate[sim, 1] <- get_miscr_by_lda(x_train, y_train, x_test, y_test, n_classes)
    
    # Regresión logística multinomial
    misc_rate[sim, 2] <- get_miscr_by_lr(x_train, y_train, x_test, y_test)
    
    # Regresión logística con LASSO
    misc_rate[sim, 3] <- get_miscr_by_lasso(x_train, y_train, x_test, y_test)
    
    # Random Forest
    misc_rate[sim, 4] <- get_miscr_by_rf(x_train, y_train, x_test, y_test)
  }
  
  print(misc_rate)
}

run_simulation(10)
