library(keras)

# generate dummy data
x_train <- matrix(runif(1000*20), nrow = 1000, ncol = 20)

y_train <- runif(1000, min = 0, max = 9) %>% 
  round() %>%
  matrix(nrow = 1000, ncol = 1) %>% 
  to_categorical(num_classes = 10)

x_test  <- matrix(runif(100*20), nrow = 100, ncol = 20)

y_test <- runif(100, min = 0, max = 9) %>% 
  round() %>%
  matrix(nrow = 100, ncol = 1) %>% 
  to_categorical(num_classes = 10)

# create model
model <- keras_model_sequential()

# define and compile the model
model %>% 
  layer_dense(units = 64, activation = 'relu', input_shape = c(20)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax') %>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_sgd(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = TRUE),
    metrics = c('accuracy')     
  )

# train
model %>% fit(x_train, y_train, epochs = 20, batch_size = 128)

# evaluate
score <- model %>% evaluate(x_test, y_test, batch_size = 128)
