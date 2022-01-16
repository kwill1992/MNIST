#MNIST dataset is available in the keras package
library(keras)
mnist <- dataset_mnist()
nx_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

save(mnist, file = "mnist.Rdata")
mnist_nocompress <- mnist
save(mnist_nocompress, file = "mnist.Rdata", compress = F)
mnist2 <- load(file = "mnist.Rdata")

saveRDS(mnist, file = "mnistRDS.Rds")
mnist.rds <- readRDS(file = "mnistRDS.Rds")

saveRDS(mnist, file = "mnistRDS_nocompress.Rds", compress = F)
mnist_nocompress.rds <- readRDS(file = "mnistRDS_nocompress.Rds")
p
