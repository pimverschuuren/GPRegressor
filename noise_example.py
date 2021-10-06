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
eps_1 = 0.
eps_2 = 0.2
eps_3 = 0.5

# Sample training data.
X_train = np.linspace(-5, 5, n_train)
Y_train = true_func(X_train)

# Define X-coordinates where the GP regressor
# is going to be evaluated. 
X_pred = np.linspace(-5, 5, n_pred)

# Evaluate the unknown function for the same 
# X-coordinates.
Y_truth = true_func(X_pred)

# Include noise in the training data.
if incl_noise:
    Y_train_2 = Y_train + np.random.normal(0., eps_2, Y_train.shape)
    Y_train_3 = Y_train + np.random.normal(0., eps_3, Y_train.shape)

Y_train_error = np.copy(Y_train)
Y_train_error.fill(eps_3)

# Instantiate a GPRegressor object with training data without any noise
# , with a little noise, with a lot of noise, and with a lot of noise but
#  including an estimate on the data errors
GP_1 = GPRegressor(X_train, Y_train)
GP_2 = GPRegressor(X_train, Y_train_2)
GP_3 = GPRegressor(X_train, Y_train_3)
GP_4 = GPRegressor(X_train, Y_train_3, Y_train_error)

# Define two sets of kernel hyperparameters for the SE kernel.
kernel_parms = [2.,2.]

# Set the kernel hyperparameters.
GP_1.set_kernel_hyperparams(kernel_parms)
GP_2.set_kernel_hyperparams(kernel_parms)
GP_3.set_kernel_hyperparams(kernel_parms)
GP_4.set_kernel_hyperparams(kernel_parms)

# Get the first prediction and covariances.
mean_1 = GP_1.get_pred(X_pred)
cov_1 = GP_1.get_cov(X_pred)

# Get the second prediction and covariances.
mean_2 = GP_2.get_pred(X_pred)
cov_2 = GP_2.get_cov(X_pred)

# Get the third prediction and covariances.
mean_3 = GP_3.get_pred(X_pred)
cov_3 = GP_3.get_cov(X_pred)

# Get the fourth prediction and covariances.
mean_4 = GP_4.get_pred(X_pred)
cov_4 = GP_4.get_cov(X_pred)


fig, axs = plt.subplots(2,2)

label_1 = "Training Data eps="+str(eps_1)
label_2 = "Training Data eps="+str(eps_2)
label_3 = "Training Data eps="+str(eps_3)
label_4 = "Training Data eps="+str(eps_3)

# Plot the GP mean functions.
axs[0,0].plot(X_pred, mean_1, '-', color='b', label="GP Prediction")
axs[0,1].plot(X_pred, mean_2, '-', color='g', label="GP Prediction")
axs[1,0].plot(X_pred, mean_3, '-', color='y', label="GP Prediction")
axs[1,1].plot(X_pred, mean_4, '-', color='orange', label="GP Prediction incl. data errors")


# Include the 1sigma variation. 
std_1 = np.sqrt(np.diag(cov_1))
std_2 = np.sqrt(np.diag(cov_2))
std_3 = np.sqrt(np.diag(cov_3))
std_4 = np.sqrt(np.diag(cov_4))

# Plot the 1sigma function variation.
axs[0,0].fill_between(X_pred, mean_1-1*std_1, mean_1+1*std_1, color='b', alpha=0.2)
axs[0,1].fill_between(X_pred, mean_2-1*std_2, mean_2+1*std_2, color='g', alpha=0.2)
axs[1,0].fill_between(X_pred, mean_3-1*std_3, mean_3+1*std_3, color='y', alpha=0.2)
axs[1,1].fill_between(X_pred, mean_4-1*std_4, mean_4+1*std_4, color='orange', alpha=0.2)

# Plot the training data.
axs[0,0].scatter(X_train, Y_train, color='black', label=label_1)
axs[0,1].scatter(X_train, Y_train_2, color='black', label=label_2)
axs[1,0].scatter(X_train, Y_train_3, color='black', label=label_3)
axs[1,1].errorbar(X_train, Y_train_3, yerr=Y_train_error, color='black', fmt='o', label=label_4)


# Plot the training data points, true unknown function and legend.
for ax in axs.flat:

    # Plot the true unknown function.
    ax.plot(X_pred, Y_truth, '--', color='r', label='True Function')

    # Set the plotting range and draw a legend.
    ax.set_ylim([-2,3])
    ax.legend(prop={"size":8})

plt.show()
