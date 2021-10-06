# GPRegressor
This repository contains a python class of which the instantiated objects can apply Gaussian Process regression. The mathematical framework behind the code is largely based on C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning. Some example scripts are included to illustrate the use of this class and the dynamics of GP regression in general.

## Dependencies
The only python packages needed to run the code are matplotlib, scipy and numpy.

# Examples
The example scripts show how one can train a GP regressor to estimate an unknown function. The unknown function is a simple sinusoidal function from which training data is sampled with an additive Gaussian noise. The GP uses a squared exponential (SE) kernel function and a zero mean function. The result is a GP posterior predictive mean that can be taken as an estimate for the unknown function. The GP posterior predictive covariance gives an estimate on the uncertainty on the function estimate.

## Kernel Hyperparameters
The hyperparameters example shows how the hyperparameters of the SE kernel influences the predictive mean and covariance. It also shows how one can use the maximum of the marginal likelihood to find an optimal point in hyperparameter space. 

## Noise
The noise example shows how noise in the data can influence the predictive mean and covariance and how they can be taken into account in the form of error bars on the training data.

## Sample functions
The sample functions example shows how one can use the predictive mean and covariance to construct a posterior from which one can sample new functions.
