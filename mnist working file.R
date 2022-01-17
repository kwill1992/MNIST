#MNIST dataset is available in the keras package
# keras library will load tensorflow also
library(keras)
# Get mnist dataset
# mnist comes as two lists.  "train" with 60,000 files and values.  "test" with 10,000 files and values.
mnist <- dataset_mnist()
#
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# This will save a compressed file.
save(mnist, file = "mnist.Rdata")
mnist_nocompress <- mnist
# This will save a non-compressed file.
save(mnist_nocompress, file = "mnist.Rdata", compress = F)
# load it back in.  But it won't load correctly because it wasn't saved as RDS file.
mnist2 <- load(file = "mnist.Rdata")

# Save and then load as compressed RDS file.
saveRDS(mnist, file = "mnistRDS.Rds")
mnist.rds <- readRDS(file = "mnistRDS.Rds")

# Save and then load as non-compress RDS file.
saveRDS(mnist, file = "mnistRDS_nocompress.Rds", compress = F)
mnist_nocompress.rds <- readRDS(file = "mnistRDS_nocompress.Rds")
p
