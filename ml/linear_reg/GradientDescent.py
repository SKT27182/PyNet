# File to calculate Gradiant Descent

import numpy as np
from typing import Union

from scipy.optimize import minimize
epsilon = 1e-10

class BaseGD:
    def __init__(self, alpha=0.01, max_iter=1000, bias=False, tol=None, penalty=None, lambda_=0.6 ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.bias = bias
        self.tol = tol
        self.penalty = penalty
        self.lambda_ = lambda_

        self.weights = None    
        self.loss_history = []

    """
    Abstract class for Gradient Descent

    Attributes
    ----------
    alpha : float
        Learning rate, default is 0.01

    max_iter : int
        Maximum number of iterations, default is 1000

    bias : bool
        If True then add bias term to X, default is False

    tol : float
        Tolerance for the cost function, default is None   
        If None then it will run for max_iter

    penalty : str
        Type of regularization, default is None

    lambda_ : float
        Regularization parameter, default is 0.6

    weights : numpy.ndarray
        Weights of the model

    loss_history : list
        List of cost function values for each iteration

    Methods
    -------

    fit(X, y)
        Fit the model

    predict(X)
        Predict the values

    r2_score(X, y)
        Calculate the r2 score

    """

    def _preprocess_input_data(self, X):
        """
        Convert input data to numpy.ndarray and reshape it if needed

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        Returns
        -------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data after preprocessing like converting to numpy.ndarray and reshaping
        """
        # if input data is not numpy.ndarray then  convert it
        if isinstance(X, np.ndarray):
            pass
        else:
            X = np.array(X)

        # if only one sample is given then reshape it
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X.astype(np.float64)

    def _bias(self, X):
        """
        Add bias term to X

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        Returns
        -------
        X : numpy.ndarray , shape (m_samples, n_features + 1 )
            Testing data with bias term
        """
        if self.bias:
            # add bias term to X (w0*x0)
            return np.insert(X, 0, 1., axis=1)
        else:
            return X

    def _weights(self, n):

        """
        Initialize weights

        Parameters
        ----------
        n : int
            Number of features

        Returns
        -------
        weights : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        """
        if self.bias:
            return np.random.uniform(low=-1, high=1, size=n +1)
            # return np.random.random(n+1)

        else:
            return np.random.uniform(low=-1, high=1, size=n)
            # return np.random.random(n)

    def _y_hat(self, X, w):

        """
        Calculate the predicted value of y

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        w : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        Returns
        -------
        y_hat : numpy.ndarray , shape (m_samples, )
            Calculated value of y
        """
        return np.dot(X, w.T)

    def _cal_loss(self, y_hat, y_true):
        """
        
        Calculate the cost function
        
        Parameters
        ----------
        y_hat : numpy.ndarray , shape (m_samples, )
            Calculated value of y
            
        y_true : numpy.ndarray , shape (m_samples, )
            True value of y
            
        Returns
        -------
        loss : float
            Loss
            
        """

        # no. of training examples
        m = len(y_true)

        # initialize loss
        total_cost = 0

        # calculate cost
        # y_hat = self._y_hat(X, w)
        total_cost = np.sum(np.square(y_hat - y_true))/(2*m)

        if self.penalty is None:
            return total_cost
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sum(np.abs(self.weights))
            return total_cost + regularization
        elif self.penalty == "l2":
            regularization = (self.lambda_ / (2*m)) * np.sum(self.weights[1:]**2)
            return total_cost + regularization
        else:
            raise ValueError("Invalid penalty type")

    def _cal_gradient(self, y_hat, y_true, X):
        """

        Calculate the gradient of the cost function to update the weights

        Parameters
        ----------
        y_hat : numpy.ndarray , shape (m_samples, )
            Calculated value of y

        y_true : numpy.ndarray , shape (m_samples, )
            True value of y

        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        Returns
        -------
        gradient : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Gradient of the cost functions

        """

        # no. of training examples
        m = len(y_true)

        grad = np.matmul((y_hat - y_true), X) / m

        if self.penalty is None:
            return grad
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sign(self.weights)
            gradient = grad + regularization
            gradient[0] = grad[0]
            return gradient
        elif self.penalty == "l2":
            regularization = (self.lambda_ / m) * self.weights
            gradient = grad + regularization
            gradient[0] = grad[0]
            return gradient
        else:
            raise ValueError("Invalid penalty type")


    def predict(self, X):
        """

        Predict the values of y for the given X

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        """
        if self.weights is None:
            raise AttributeError("You have to fit the model first")

        X = self._preprocess_input_data(X)
        X = self._bias(X)
        return self._y_hat(X, self.weights)

    def r2_score(self, X, y):
        """

        Calculate the R2 score

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        y : numpy.ndarray , shape (m_samples, )
            True values

        Returns
        -------
        score : float
            R2 score

        """
        y_hat = self.predict(X)
        SSres = np.sum((y - y_hat)**2)
        SStot = np.sum((y - y.mean())**2)

        return 1 - (SSres / SStot)

class BatchGD(BaseGD):

    def __init__(self, alpha, max_iter, bias=False, tol=None, penalty=None, lambda_=0.6 ):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)




    def fit(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values
        alpha : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        tol : float, default None
            Tolerance for stopping criteria

        Returns
        -------
        None, just updates the weights and loss_history attributes of the class

        """

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # to include the bias term or not and initialize weights
        X = self._bias(X)
        self.weights = self._weights(n)

        # iterate until max_iter

        for i in range(self.max_iter):

            # calculate the gradient of Loss/Cost Function

            y_hat = self._y_hat(X, self.weights)
            gradient = self._cal_gradient(y_hat, y, X)

            # update the weights
            self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            y_hat = self._y_hat(X, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y))

            # Break the loop if loss is not changing much
            if i > 0 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(
                    f"Loss is nan at iteration {i}. Hence, stopping the training")
                break

class MiniBatchGD(BaseGD):
    def __init__(self, alpha, max_iter, bias=False, tol=None, penalty=None, lambda_=0.6 ):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)


    def fit(self, X, y, batch_size=32):

        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values
        alpha : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        tol : float, default None
            Tolerance for stopping criteria

        Returns
        -------
        None, just updates the weights and loss_history attributes of the class

        """

        self.batch_size = batch_size

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # no. of samples
        m = X.shape[0]

        # to include the bias term or not and initialize weights
        X = self._bias(X)
        self.weights = self._weights(n)

        # converting y to numpy array b/c if it is dataframe then it will give error while creating batches
        y = np.array(y)

        for i in range(self.max_iter):

            # shuffle the data
            shuffle_indices = np.random.permutation(np.arange(m))
            X_shuffle = X[shuffle_indices]
            y_shuffle = y[shuffle_indices]

            # split the data into batches
            for j in range(0, m, self.batch_size):
                X_batch = X_shuffle[j:j + self.batch_size]
                y_batch = y_shuffle[j:j + self.batch_size]

                # calculate the gradient of Loss/Cost Function
                y_hat = self._y_hat(X_batch, self.weights)
                gradient = self._cal_gradient(y_hat, y_batch, X_batch)

                # update the weights
                self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            y_hat = self._y_hat(X, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y))

            # Break the loop if loss is not changing much
            if i > 1 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(
                    f"Loss is nan at iteration {i}. Hence, stopping the training")
                break

class StochasticGD(BaseGD):
    def __init__(self, alpha, max_iter, bias=False, tol=None, penalty=None, lambda_=0.6 ):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)


    def fit(self, X, y):

        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values
        alpha : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        tol : float, default None
            Tolerance for stopping criteria

        Returns
        -------
        None, just updates the weights and loss_history attributes of the class

        """

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # no. of samples
        m = X.shape[0]

        # to include the bias term or not and initialize weights
        X = self._bias(X)
        self.weights = self._weights(n)

        # converting y to numpy array b/c if it is dataframe then it will give error while creating batches
        y = np.array(y)

        for i in range(self.max_iter):

            # Update the weights one by one for each data point
            for j in range(0, m):

                # take a data point randomly at a time and update the weights
                index = np.random.randint(m)

                X_ = X[index:index + 1]
                y_ = y[index:index + 1]

                # calculate the gradient of Loss/Cost Function
                y_hat = self._y_hat(X_, self.weights)
                gradient = self._cal_gradient(y_hat, y_, X_)

                # update the weights
                self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            y_hat = self._y_hat(X_, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y_))

            # Break the loop if loss is not changing much
            if i > 1 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(f"Loss is nan at iteration {i}. Hence, stopping the training")
                break

class LinearSearchGD(BaseGD):
    def __init__(self, max_iter, bias=False, tol=None,  penalty=None, lambda_=0.1):
        super().__init__(alpha=None, max_iter=max_iter, bias=bias, tol=tol, penalty=penalty, lambda_=lambda_)
        self.alpha_history = []

    """
    Linear Search Gradient Descent
    

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations
    bias : bool
        To include the bias term or not
    tol : float
        Tolerance for the stopping criterion
   
    Attributes
    ----------
    weights : numpy.ndarray , shape (n_features, )
        Weights after fitting the model
    loss_history : list
        Loss history after each iteration



    Methods
    -------
    fit(X, y)
        Fit the model according to the given training data.

    predict(X)
        Predict the y values for the given X.

    r2_score(y_true, y_pred)
        Calculate the R2 score.

    """

    def phi(self, X, y, weights, alpha):
        # calculate the loss function w.r.t alpha

        y_hat = self._y_hat(X, weights)
        grad = self._cal_gradient(y_hat, y, X)

        weights_new = weights - alpha * grad
        y_hat_new = self._y_hat(X, weights_new)

        return self._cal_loss(y_hat_new, y)

    def phi_prime(self, X, y, weights, alpha):
        # calculate the derivative of phi w.r.t alpha using central difference method
        h = 1e-6
        phi_plus = self.phi(X, y, weights, alpha + h)
        phi_minus = self.phi(X, y, weights, alpha - h)
        return (phi_plus - phi_minus) / (2 * h)

    def secant(self, X, y, weights, alpha, alpha_prev):

        phi_prime_curr = self.phi_prime(X, y, weights, alpha)
        phi_prime_prev = self.phi_prime(X, y, weights, alpha_prev)

        new_alpha = alpha - \
            (((alpha - alpha_prev) / (phi_prime_curr -
             phi_prime_prev + 1e-10)) * phi_prime_curr)

        alpha, alpha_prev = new_alpha, alpha

        # if abs(self.phi(X, y, weights, new_alpha) - phi_prime_curr) < 1e-4:
        if abs(new_alpha - alpha_prev) < 1e-8:
            return new_alpha
        else:
            return self.secant(X, y, weights, alpha, alpha_prev)

    def optimize_alpha(self, X, y, weights):
        alpha = 0.01
        alpha_prev = 0.1
        new_alpha = self.secant(X, y, weights, alpha, alpha_prev)
        if new_alpha < 1e-4:
            new_alpha = 1e-4
        elif new_alpha > 0.9:
            new_alpha = 0.9
        return new_alpha

    def fit(self, X, y, alpha_callback=None):
        X = self._preprocess_input_data(X)
        self.weights = self._weights(X.shape[1])
        X = self._bias(X)

        for i in range(self.max_iter):

            weight = self.weights

            alpha = self.optimize_alpha(X, y, weight)

            if alpha_callback is not None:
                alpha_callback(alpha)

            y_hat = self._y_hat(X, self.weights)
            self.gradient = self._cal_gradient(y_hat, y, X)

            self.weights = self.weights - alpha * self.gradient

            # keeping track of cost/loss
            y_hat = self._y_hat(X, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y))

            # Break the loop if loss is not changing much
            if i > 0 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    print(
                        f"Loss is not changing much at iteration {i}. Hence, stopping the training")
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(
                    f"Loss is nan at iteration {i}. Hence, stopping the training")
                break
