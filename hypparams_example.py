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
eps = 0.1

# Sample training data.
X_train = np.linspace(-5, 5, n_train)
Y_train = true_func(X_train)

# Include error on the data.
Y_train_error = np.copy(Y_train)
Y_train_error.fill(eps)

# Define X-coordinates where the GP regressor
# is going to be evaluated. 
X_pred = np.linspace(-5, 5, n_pred)

# Evaluate the unknown function for the same 
# X-coordinates.
Y_truth = true_func(X_pred)

# Include noise in the training data.
if incl_noise:
    Y_train = Y_train + np.random.normal(0., eps, Y_train.shape)

# Instantiate a GPRegressor object.
GP = GPRegressor(X_train, Y_train, Y_train_error)

# Define two sets of kernel hyperparameters for the SE kernel.
kernel_parms_1 = [1.,2.]
kernel_parms_2 = [1,0.1]
kernel_parms_3 = [2,2]

# Set the kernel hyperparameters.
GP.set_kernel_hyperparams(kernel_parms_1)

# Get the first prediction and covariances.
mean_1 = GP.get_pred(X_pred)
cov_1 = GP.get_cov(X_pred)

# Set the kernel hyperparameters.
GP.set_kernel_hyperparams(kernel_parms_2)

# Get the second prediction and covariances.
mean_2 = GP.get_pred(X_pred)
cov_2 = GP.get_cov(X_pred)

# Set the kernel hyperparameters.
GP.set_kernel_hyperparams(kernel_parms_3)

# Get the third prediction and covariances.
mean_3 = GP.get_pred(X_pred)
cov_3 = GP.get_cov(X_pred)

# Define parameter starting values and bounds for the 
# maximization of the marginal likelihood.
kernel_parm_start = [1, 2]
kernel_parm_bounds = [[0.001,100000], [0.001, 100000]]

# Find an optimal set of kernel hyperparameters by maximizing 
# the marginal likelihood.
kernel_parms_4 = GP.maximizeMargLikelihood(kernel_parm_start, kernel_parm_bounds)

# Set the kernel hyperparameters.
GP.set_kernel_hyperparams(kernel_parms_4)


# Get the fourth prediction and covariances.
mean_4 = GP.get_pred(X_pred)
cov_4 = GP.get_cov(X_pred)


fig, axs = plt.subplots(2,2)

label_1 = "GP Prediction (A,l)=("+str(kernel_parms_1[0])+","+str(kernel_parms_1[1])+")"
label_2 = "GP Prediction (A,l)=("+str(kernel_parms_2[0])+","+str(kernel_parms_2[1])+")"
label_3 = "GP Prediction (A,l)=("+str(kernel_parms_3[0])+","+str(kernel_parms_3[1])+")"
label_4 = "GP Prediction (A,l)=("+str(round(kernel_parms_4[0],3))+","+str(round(kernel_parms_4[1],3))+")"

# Plot the GP mean functions.
axs[0,0].plot(X_pred, mean_1, '-', color='b', label=label_1)
axs[0,1].plot(X_pred, mean_2, '-', color='g', label=label_2)
axs[1,0].plot(X_pred, mean_3, '-', color='y', label=label_3)
axs[1,1].plot(X_pred, mean_4, '-', color='orange', label=label_4)


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

# Plot the training data points, true unknown function and legend.
for ax in axs.flat:

    # Plot the training data.
    ax.scatter(X_train, Y_train, color='black', label='Training Data') 

    # Plot the true unknown function.
    ax.plot(X_pred, Y_truth, '--', color='r', label='True Function')

    # Set the plotting range and draw a legend.
    ax.set_ylim([-2,3])
    ax.legend(prop={"size":8})

plt.show()
