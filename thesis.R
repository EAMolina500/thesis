library(MASS)

# generating random vectors

# Generating Random Numbers from a Uniform Distribution
random_uniform <- runif(1)  # Generates one random number between 0 and 1
#print(random_uniform)

# Generating Random Numbers from a Normal Distribution
random_normal <- rnorm(1, mean = 0, sd = 1) 
#print(random_normal)

# Generating Random Integers
random_integer <- sample(1:100, 1)  # Generates one random integer between 1 and 100
#print(random_integer)

# Generating Random Numbers from a Binomial Distribution
random_binomial <- rbinom(1, size = 10, prob = 0.5)  
#print(random_binomial)

# Generate multiple random numbers from a binomial distribution
random_binomials <- rbinom(5, size = 10, prob = 0.5) 
#print(random_binomials)

gen_mean_list <- function(num_means, p, min = 1, max = 100) {
  mean_list <- vector("list", num_means)
  for (i in 1:num_means) {
    mean_list[[i]] <- runif(p, min = min, max = max)
  }
  return(mean_list)
}

gen_cov_matrix <- function(p) {
  sigma <- matrix(0, p, p)
  diag(sigma) <- 1
  
  for (row in 1:p) {
    for (col in 1:p) {
      if (row != col) {
        sigma[row, col] <- runif(1, min = -1, max = 1)
      }
    }
  }
  
  # make matrix symetric
  sigma <- (sigma + t(sigma)) / 2
  
  # make matrix positive definite
  #sigma <- sigma + p * diag(p)
  diag(sigma) <- diag(sigma) + 1e-6
  
  return(sigma)
}

gen_data <- function(n, mean_list, sigma, matrices_num = 1) {
  data_list <- vector("list", matrices_num)
  
  for (i in 1:matrices_num) {
    data_list[[i]] <- mvrnorm(n, mean_list[[i]], sigma)
  }
  
  return(data_list)
}

gen_x_data <- function(matrix_list) {
  x_data <- do.call(rbind, matrix_list)
  return(x_data)
}

gen_y_data <- function(matrix_list) {
  m_num <- length(matrix_list)
  m <- nrow(matrix_list[[1]])
  y_data <- factor(rep(1:m_num, each = m))
  return(y_data)
}

simulate <- function(sim_num = 1) {
  for (s in 1:sim_num) {
    p <- 20
    n_train <- 15 
    n_test <- 200
    mean_list <- gen_mean_list(3, p)
    sigma <- gen_cov_matrix(p)
    
    train_data <- gen_data(n_train, mean_list, sigma, 3)
    test_data <- gen_data(n_test, mean_list, sigma, 3)
    
    x_train <- gen_x_data(train_data)
    y_train <- gen_y_data(train_data)
    
    x_test <- gen_x_data(test_data)
    y_test <- gen_y_data(test_data)
    
    print(x_train)
  }
}

  
  
  
  
  
  
  
  
  