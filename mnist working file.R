#MNIST dataset is available in the keras package

# Basic Keras from here:
# https://cran.r-project.org/web/packages/keras/vignettes/index.html

# Keras or Tensorflow?
https://www.simplilearn.com/keras-vs-tensorflow-vs-pytorch-article
# next one is very good
https://www.guru99.com/tensorflow-vs-keras.html


# try with Keras, Tensorflow, caffe, PyTorch, CNTK, Theano, dataiku, MXNet, magick, imager
# Get working with Python also
# get working by calling python in R


#check and install packages if needed.  Need pacman installed.
library(pacman)
#pacman::p_load(ggplot2, tidyr, dplyr)

# keras library will load tensorflow also
#The Keras R interface uses the TensorFlow backend engine by default.
pacman::p_load(keras)
#library(keras)
# Get mnist dataset
# mnist comes as two lists.  "train" with 60,000 files and values.  "test" with 10,000 files and values.
#mnist <- dataset_mnist()
#

# This will save a compressed file.
#save(mnist, file = "mnist.Rdata")
#mnist_nocompress <- mnist
# This will save a non-compressed file.
#save(mnist_nocompress, file = "mnist.Rdata", compress = F)
# load it back in.  But it won't load correctly because it wasn't saved as RDS file.
#mnist2 <- load(file = "mnist.Rdata")

# Save and then load as compressed RDS file.
#saveRDS(mnist, file = "mnistRDS.Rds")
#mnist.rds <- readRDS(file = "mnistRDS.Rds")

# Save and then load as non-compress RDS file.
#saveRDS(mnist, file = "mnistRDS_nocompress.Rds", compress = F)
#mnist_nocompress.rds <- readRDS(file = "mnistRDS_nocompress.Rds")

# Preparing the Data
mnist <- readRDS(file = "mnistRDS.Rds")

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_train)

# The x data is a 3-d array (images,width,height) of grayscale values . 
# To prepare the data for training we convert the 3-d arrays into matrices by 
# reshaping width and height into a single dimension (28x28 images are flattened 
#                                                     into length 784 vectors). T
# hen, we convert the grayscale values from integers ranging between 0 to 255 into 
# floating point values ranging between 0 and 1:
  
# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# The y data is an integer vector with values ranging from 0 to 9. 
# To prepare this data for training we one-hot encode the vectors into 
# binary class matrices using the Keras to_categorical() function:
  
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Defining the Model
# 
# The core data structure of Keras is a model, a way to organize layers. 
# The simplest type of model is the Sequential model, a linear stack of layers.
# 
# We begin by creating a sequential model and then adding layers using the pipe (%>%) operator:
  
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
# The input_shape argument to the first layer specifies the shape of the input data (a length 784 numeric vector representing a grayscale image). The final layer outputs a length 10 numeric vector (probabilities for each digit) using a softmax activation function.
# 
# Use the summary() function to print the details of the model:
  
summary(model)

# Next, compile the model with appropriate loss function, optimizer, and metrics:
  
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
  )

# Training and Evaluation
# 
# Use the fit() function to train the model for 30 epochs using batches of 128 images:
  
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
  )

#The history object returned by fit() includes loss and accuracy metrics which we can plot:
  
plot(history)

#Evaluate the model’s performance on the test data:
  
model %>% evaluate(x_test, y_test)
#Generate predictions on new data:
  
model %>% predict_classes(x_test)
