import numpy as np
from scipy.optimize import minimize

class GPRegressor:

    """A class that can train a GP Regressor, optimize its
       hyperparameters and make predictions"""
    
    def __init__(self, X_train, Y_train, Y_train_error = []):
        """Initialize the regressor with training data
        
        Input
        ------
        X_train: The X-coordinates of the training data points.
        Y_train: The Y-coordinates of the training data points.
        """

        self.X_train = X_train
        self.Y_train = Y_train
        self.Y_train_cov = np.zeros(Y_train.shape)

        if (len(Y_train_error) > 0):
            Y_train_cov = np.power(Y_train_error,2)
            self.Y_train_cov = np.diag(Y_train_cov)

        # Place holder for the kernel hyperparameters
        self.kernel_hyp = []


    # Squared Exponential Kernel
    # Hyperparameters
    # ---------------
    # A: Amplitude of the correlations
    # l: Length scale of correlations
    def kernel(self, X1, X2, A = 1, l = 2):
        
        if len(self.kernel_hyp) > 0:
            A = self.kernel_hyp[0]
            l = self.kernel_hyp[1]

        K = np.empty((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = A*np.exp(-1/float(l) * np.linalg.norm([x1 - x2], 2)**2)
        return K

    
    # Get the mean function as GP regressor prediction given a set of X-coordinates.
    # Definition from C. E. Rasmussen, C. K. I. Williams, Gaussian Processes for
    # Machine Learning, The MIT Press, 2006
    def get_pred(self, X_pred):

        # Get the kernel matrices.
        K_tt = self.kernel(self.X_train,  self.X_train)
        K_st = self.kernel(X_pred, self.X_train)

        # Return the mean function.
        mean = (K_st.dot(np.linalg.inv(K_tt + self.Y_train_cov))).dot(self.Y_train)

        return mean

    def get_kernel_parms(self):
        return self.kernel_hyp

    # Get the covariance given a set of X-coordinates.
    # Definition from C. E. Rasmussen, C. K. I. Williams, Gaussian Processes for
    # Machine Learning, The MIT Press, 2006
    def get_cov(self, X_pred):
        
        K_st = self.kernel(X_pred,  self.X_train)
        K_tt = self.kernel(self.X_train, self.X_train)
        K_ss = self.kernel(X_pred,  X_pred)
        K_ts = self.kernel(self.X_train, X_pred)

        cov  = K_ss - ((K_st.dot(np.linalg.inv(K_tt + self.Y_train_cov)).dot(K_ts)))
        
        return cov

    # Set the values of the kernel parameters.
    def set_kernel_hyperparams(self, hyperparams):
        
        self.kernel_hyp = []

        for i in range(len(hyperparams)):            
            self.kernel_hyp.append(hyperparams[i])

    # Negative marginal likelihood that can be minimized w.r.t.
    # the kernel hyperparameters.
    # Definition from C. E. Rasmussen, C. K. I. Williams, Gaussian Processes for
    # Machine Learning, The MIT Press, 2006
    def negLogMargLikelihood(self, hyperparams):

        self.kernel_hyp = hyperparams
        
        K = self.kernel(self.X_train, self.X_train)

        negMargLLH = (self.Y_train.dot(np.linalg.inv(K + self.Y_train_cov))).dot(self.Y_train) + np.log(np.linalg.det(K + self.Y_train_cov))
        
        return negMargLLH

    # Maximize the marginal likelihood i.e. minimize the negative log marginal likelihood to find optimal values for the kernel parameters.
    def maximizeMargLikelihood(self, param_start, param_bounds):
    
        fit_result = minimize(self.negLogMargLikelihood, param_start, method='L-BFGS-B', bounds=param_bounds)

        return fit_result.x
  
    # Sample a new function from the GP posterior distribution given
    # a mean and covariance.
    def sample_posterior(self, mean, cov):
        return np.random.multivariate_normal(mean, cov)

