# Limpiar el entorno de trabajo
rm(list = ls())

# Cargar las librerías necesarias
library(Matrix)
library(MASS) # LDA
library(randomForest)
library(plyr) # ddply
library(tidyverse)
library(glmnet) # Regresión logística
library(SplitGLM)
library(mvnfast)

# Definir el número de variables
p = 200

# Definir los vectores de medias para las tres clases
###
#  Las medias abajo definen el promedio de cada clase. Medias cercanas reflejan
#  -parecidos-, es decir, pasto y gramilla por ejemplo. Medias lejanas reflejan
#  lo contrario, pasto y agua por ejemplo.
###
m1 = c(1,  2, 3, 5, 1.5, rep(0, p - 5))
m2 = c(25, 74, 42, 27, 29, rep(0, p - 5))
m3 = c(4, 7, 1, rep(0, p - 3))

# Definir la matriz de covarianza
Sigma <- matrix(0, p, p)
Sigma[1:5, 1:5] <- 0.5 
diag(Sigma) <- 1

# Calcular los coeficientes beta para las clases
beta1 <- (m1 - m3) %*% solve(Sigma)
beta2 <- (m2 - m3) %*% solve(Sigma)

# Definir el número de repeticiones para la simulación
R = 10
mis_cla_rate = array(0, dim = c(R, 5))
t0 <- Sys.time()  # Tiempo inicial
set.seed(2)

# Bucle principal para realizar las simulaciones
for (r in 1:R) {
  
  # Generar datos de entrenamiento
  n = 150
  dat1.train <- mvrnorm(n, m1, Sigma)
  dat2.train <- mvrnorm(n, m2, Sigma)
  dat3.train <- mvrnorm(n, m3, Sigma)
  x.train <- rbind(dat1.train, dat2.train, dat3.train)
  y.train = factor(rep(1:3, c(n, n, n)))
  
  # Generar datos de prueba
  m = 2000
  dat1.test <- mvrnorm(m, m1, Sigma)
  dat2.test <- mvrnorm(m, m2, Sigma)
  dat3.test <- mvrnorm(m, m3, Sigma)
  x.test <- rbind(dat1.test, dat2.test, dat3.test)
  y.test = factor(rep(1:3, c(m, m, m)))
  
  # LDA (Análisis Discriminante Lineal)
  zlda = lda(x.train, y.train, prior = c(1, 1, 1) / 3, method = "mle")
  predicted.classes = predict(zlda, x.test)$class
  mis_cla_rate[r, 1] <- mean(y.test != predicted.classes) # Tasa de error
  
  # Regresión logística multinomial
  ml <- glmnet(x.train, as.factor(y.train), alpha = 1, family = "multinomial", lambda = 0)
  predicted.classes = predict(ml, x.test, s = 0.0001, type = "class")
  mis_cla_rate[r, 2] = mean(y.test != predicted.classes) # Tasa de error
  
  # Regresión logística con LASSO
  cv.lasso <- cv.glmnet(x.train, y.train, alpha = 1, family = "multinomial")
  l = cv.lasso$lambda.min
  model <- glmnet(x.train, y.train, alpha = 1, family = "multinomial", lambda = l, standardized = FALSE)
  predicted.classes = predict(model, x.test, s = 0.0001, type = "class")
  mis_cla_rate[r, 3] <- mean(y.test != predicted.classes) # Tasa de error
  
  # Random Forest
  rf = randomForest(x.train, y.train, importance = TRUE)
  predicted.classes = predict(rf, x.test, type = "class")
  mis_cla_rate[r, 4] <- mean(y.test != predicted.classes) # Tasa de error
  
  # Split Regression
  y.train = rep(1:0, c(n, n))
  x.train1 = rbind(dat1.train, dat3.train)
  split.out1 <- cv.SplitGLM(x.train1, y.train, glm_type = "Logistic", G = 10, include_intercept = TRUE, alpha_s = 3/4, alpha_d = 1, n_lambda_sparsity = 50, n_lambda_diversity = 50, tolerance = 1e-3, max_iter = 1e3, n_folds = 5, active_set = FALSE, n_threads = 1)
  split.coef1 <- as.matrix(coef(split.out1))
  
  x.train2 = rbind(dat2.train, dat3.train)
  split.out2 <- cv.SplitGLM(x.train2, y.train, glm_type = "Logistic", G = 10, include_intercept = TRUE, alpha_s = 3/4, alpha_d = 1, n_lambda_sparsity = 50, n_lambda_diversity = 50, tolerance = 1e-3, max_iter = 1e3, n_folds = 5, active_set = FALSE, n_threads = 1)
  split.coef2 <- as.matrix(coef(split.out2))
  
  # Predicciones
  prob = array(0, dim = c(3 * m, 3))
  prob[, 1] = exp(split.coef1[1] + x.test %*% split.coef1[2:(p + 1), 1]) / (1 + exp(split.coef1[1] + x.test %*% split.coef1[2:(p + 1), 1]) + exp(split.coef2[1] + x.test %*% split.coef2[2:(p + 1), 1]))
  prob[, 2] = exp(split.coef2[1] + x.test %*% split.coef2[2:(p + 1), 1]) / (1 + exp(split.coef1[1] + x.test %*% split.coef1[2:(p + 1), 1]) + exp(split.coef2[1] + x.test %*% split.coef2[2:(p + 1), 1]))
  prob[, 3] = 1 - (prob[, 1] + prob[, 2])
  
  y.pred = array(0, dim = c(3 * m))
  for (i in 1:(3 * m)) {
    y.pred[i] = which.max(prob[i, ])
  }
  
  mis_cla_rate[r, 5] <- mean(y.test != y.pred) # Tasa de error
}

t1 <- Sys.time()  # Tiempo final
tt <- t1 - t0

# Mostrar el tiempo de ejecución total
print(tt)

# Visualizar los resultados
datos = stack(data.frame(mis_cla_rate))
boxplot(datos$values ~ datos$ind)
