rm(list = ls())
library(keras3)
set_random_seed(1)
x <- cbind(rbeta(10000,3,2), rbeta(10000,2,5))
y <- rnorm(10000,mean = x[,1]^2-3*x[,2]+5,sd = 2*x[,1])
x_test <- cbind(rbeta(1000,3,2), rbeta(1000,2,5))
y_test <- rnorm(1000,mean = x_test[,1]^2-3*x_test[,2]+5,sd = mean(x[,1]))
hist(y)

input1 <- keras_input(shape=dim(x)[2], name = 'covariates')
x_1 <- layer_dense(input1, units = 12, activation = 'relu')

x_2 <- layer_dense(x_1, 12, activation = 'relu')

mu <- layer_dense(x_2, 1, activation = 'linear', name = "mean")
sig <- layer_dense(x_2, 1, activation = 'exponential', name = "sigma")

out_concat <- layer_concatenate(mu,sig)
out_concat <- layer_identity(out_concat, name='params')
# out_concat <- layer_identity(probs, name='outs')
model <- keras_model(inputs = list(input1), 
                     outputs = out_concat, name = "norm_dist")
summary(model)


nloglik_loss_normal  = function (y_true, y_pred){
    # print(numbasis)
    mu <- y_pred[,1]
    sig <- y_pred[,2]
    isthisloss <- op_sum(op_log(sig) + 0.5*((y_true-mu)/sig)**2) 
    return(isthisloss)
}

model |> compile(
    loss = nloglik_loss_normal,
    optimizer = optimizer_adam(learning_rate=0.01)
)

history <- model |> fit(
    x= x,
    y=y,
    epochs = 100,
    batch_size = 128,
    callbacks=list(callback_early_stopping(monitor = "val_loss",
                                                     min_delta = 0, patience = 10)),
    validation_split = 0.2
)

tmp <- as.matrix(model(x_test))
true_mean <- x_test[,1]^2-3*x_test[,2]+5
true_sd <- 2*x_test[,1]

plot(tmp[,1],true_mean)
abline(0,1)

plot(tmp[,2],true_sd)
abline(0,1)


file_url <- "https://github.com/reetamm/AI4stats/blob/main/weather.RDS?raw=true"
weather <- readRDS(url(file_url))
weather <- weather[weather$month==7 & weather$loc==1,]
head(weather)

mnth <- weather$month
tmax <- weather$tmax - 273 #convert tmax to celsius
pr <- log(weather$pr + 0.0001) #convert pr to log-scale
plot(tmax,pr,pch=20)
n_total <- length(tmax)
train_ind <- sample(1:n_total,ceiling(0.8*n_total)) #80% training data

tmax_range <- range(tmax)
y1 <- tmax
X <- cbind(1,pr)

# train and validation data
y1_train <- y1[train_ind]
y1_test <- y1[-train_ind]

# For unconditional density, with only intercept
X1_train <- matrix(X[train_ind,1],ncol = 1)
X1_test <- matrix(X[-train_ind,1],ncol = 1)

# For conditional density of tmax with intercept and log-pr
X2_train <- X[train_ind,1:2]
X2_test <- X[-train_ind,1:2]


input1 <- keras_input(shape=dim(x)[2], name = 'covariates')
x_1 <- layer_dense(input1, units = 12, activation = 'relu')

x_2 <- layer_dense(x_1, 12, activation = 'relu')

mu <- layer_dense(x_2, 1, activation = 'linear', name = "mean")
sig <- layer_dense(x_2, 1, activation = 'exponential', name = "sigma")

out_concat <- layer_concatenate(mu,sig)
out_concat <- layer_identity(out_concat, name='params')
# out_concat <- layer_identity(probs, name='outs')
model <- keras_model(inputs = list(input1), 
                     outputs = out_concat, name = "norm_dist")
summary(model)

model |> compile(
    loss = nloglik_loss_normal,
    optimizer = optimizer_adam(learning_rate=0.01)
)

history <- model |> fit(
    x= X2_train,
    y=y1_train,
    epochs = 100,
    batch_size = 128,
    callbacks=list(callback_early_stopping(monitor = "val_loss",
                                           min_delta = 0, patience = 10)),
    validation_split = 0.2
)

tmp <- as.matrix(model(X2_test))

head(y_test)

y_pred <- matrix(NA,1000,6)
for(i in 1:6)
    y_pred[,i] <- rnorm(1000,tmp[i,1],tmp[i,2])
par(mfrow=c(2,3))
for(i in 1:6){
    prcp <- round(exp(X2_test[i,2]) - 0.0001,2)
    plot(density(y_pred[,i]),main=paste0('prcp = ',prcp))
    abline(v=y1_test[i])  
}
par(mfrow=c(1,1))
