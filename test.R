library(MASS)
library(glmnet)
library(randomForest)

# Genera una lista de medias aleatorias para cada clase
gen_mean_list <- function(num_classes, num_features, min_value = 1, max_value = 100) {
  means <- vector("list", num_classes)
  for (i in 1:num_classes) {
    means[[i]] <- runif(num_features, min = min_value, max = max_value)
  }
  return(means)
}

# Genera una matriz de covarianza positiva definida
gen_cov_matrix <- function(num_features) {
  # Crea una matriz aleatoria
  random_matrix <- matrix(runif(num_features^2, min = -1, max = 1), num_features, num_features)
  
  # Hacer la matriz simétrica
  cov_matrix <- (random_matrix + t(random_matrix)) / 2
  
  # Garantiza que la matriz sea positiva definida
  cov_matrix <- cov_matrix %*% t(cov_matrix) + 1e-6 * diag(num_features)
  
  return(cov_matrix)
}

# Genera datos multivariados normales con medias y matriz de covarianza especificadas
gen_data <- function(num_samples, means, cov_matrix, num_classes) {
  data <- vector("list", num_classes)
  
  for (i in 1:num_classes) {
    data[[i]] <- mvrnorm(num_samples, means[[i]], cov_matrix)
  }
  
  return(data)
}

# Combina las matrices de datos en un solo marco de datos
combine_x_data <- function(data_list) {
  combined_data <- do.call(rbind, data_list)
  return(combined_data)
}

# Crea las etiquetas de clase para los datos
create_y_labels <- function(data_list) {
  num_classes <- length(data_list)
  num_samples_per_class <- nrow(data_list[[1]])
  labels <- factor(rep(1:num_classes, each = num_samples_per_class))
  return(labels)
}

# Función principal para ejecutar la simulación
run_simulation <- function(num_simulations = 1) {
  mis_cla_rate <- matrix(0, nrow = num_simulations, ncol = 4) # Tasa de error para cada método
  
  for (sim in 1:num_simulations) {
    num_features <- 20
    num_train_samples <- 15 
    num_test_samples <- 200
    num_classes <- 3
    
    means <- gen_mean_list(num_classes, num_features)
    cov_matrix <- gen_cov_matrix(num_features)
    
    train_data <- gen_data(num_train_samples, means, cov_matrix, num_classes)
    test_data <- gen_data(num_test_samples, means, cov_matrix, num_classes)
    
    x_train <- combine_x_data(train_data)
    y_train <- create_y_labels(train_data)
    
    x_test <- combine_x_data(test_data)
    y_test <- create_y_labels(test_data)
    
    # LDA (Análisis Discriminante Lineal)
    lda_model <- lda(x_train, y_train)
    lda_pred <- predict(lda_model, x_test)$class
    mis_cla_rate[sim, 1] <- mean(y_test != lda_pred) # Tasa de error LDA
    
    # Regresión logística multinomial
    logit_model <- glmnet(x_train, as.factor(y_train), alpha = 1, family = "multinomial", lambda = 0)
    logit_pred <- predict(logit_model, x_test, s = 0.0001, type = "class")
    mis_cla_rate[sim, 2] <- mean(y_test != logit_pred) # Tasa de error Regresión Logística Multinomial
    
    # Regresión logística con LASSO
    cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "multinomial")
    lasso_lambda <- cv_lasso$lambda.min
    lasso_model <- glmnet(x_train, y_train, alpha = 1, family = "multinomial", lambda = lasso_lambda)
    lasso_pred <- predict(lasso_model, x_test, s = 0.0001, type = "class")
    mis_cla_rate[sim, 3] <- mean(y_test != lasso_pred) # Tasa de error LASSO
    
    # Random Forest
    rf_model <- randomForest(x_train, y_train)
    rf_pred <- predict(rf_model, x_test, type = "class")
    mis_cla_rate[sim, 4] <- mean(y_test != rf_pred) # Tasa de error Random Forest
    
    # Imprime las tasas de error para esta simulación
    cat(sprintf("Simulation %d\n", sim))
    cat("LDA Error Rate: ", mis_cla_rate[sim, 1], "\n")
    cat("Logistic Regression Error Rate: ", mis_cla_rate[sim, 2], "\n")
    cat("LASSO Error Rate: ", mis_cla_rate[sim, 3], "\n")
    cat("Random Forest Error Rate: ", mis_cla_rate[sim, 4], "\n\n")
  }
  
  return(mis_cla_rate)
}

# Ejecuta la simulación
set.seed(2)
results <- run_simulation(num_simulations = 10)
