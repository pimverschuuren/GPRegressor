from GPRegressor import *

import matplotlib.pyplot as plt

# A function that the GP regressor tries to estimate and
# from which data is sampled. In practice, this function is unknown.
def true_func(x):
    return np.sin(2 * np.pi * x / 6)

# Define the number of data points used to train the GP regressor.
n_train = 7

# Define the number of data points used to evaluate the trained GP regressor.
n_pred = 100

# Add noise to the sampled training data.
incl_noise = True

# Try out different values of eps to see the effect of noise in
# data on the GP regressor.
eps = 0.1

# Sample training data.
X_train = np.linspace(-5, 5, n_train)
Y_train = true_func(X_train)

# Define X-coordinates where the GP regressor
# is going to be evaluated. 
X_pred = np.linspace(-5, 5, n_pred)

# Evaluate the unknown function for the same 
# X-coordinates.
Y_truth = true_func(X_pred)

# Instantiate a GPRegressor object with training data without any noise
# , with a little noise, with a lot of noise, and with a lot of noise but
#  including an estimate on the data errors
GP = GPRegressor(X_train, Y_train)

# Define two sets of kernel hyperparameters for the SE kernel.
kernel_parms = [2.,2.]

# Set the kernel hyperparameters.
GP.set_kernel_hyperparams(kernel_parms)

# Get the first prediction and covariances.
mean = GP.get_pred(X_pred)
cov = GP.get_cov(X_pred)

# Sample new functions from the posterior given
# the mean and covariance.
sampled_func_1 = GP.sample_posterior(mean, cov)
sampled_func_2 = GP.sample_posterior(mean, cov)
sampled_func_3 = GP.sample_posterior(mean, cov)

fig, ax = plt.subplots()

# Plot the GP mean functions.
ax.plot(X_pred, mean, '-', color='b', label="GP Prediction")

# Include the 1sigma variation. 
std = np.sqrt(np.diag(cov))

ax.fill_between(X_pred, mean-1*std, mean+1*std, color='b', alpha=0.2)

ax.scatter(X_train, Y_train, color='black', label="Training data")

ax.plot(X_pred, sampled_func_1, '-', color='g', label="Sampled function 1")
ax.plot(X_pred, sampled_func_2, '-', color='orange', label="Sampled function 2")
ax.plot(X_pred, sampled_func_3, '-', color='darkviolet', label="Sampled function 3")

# Plot the training data points, true unknown function and legend.
ax.plot(X_pred, Y_truth, '--', color='r', label='True Function')
ax.set_ylim([-2,3])
ax.legend(prop={"size":8})

plt.show()
