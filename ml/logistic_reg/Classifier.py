# Classes for Binary and Multiclass Classification

import numpy as np
epsilon = 1e-10

class BinaryClassifier:
    def __init__(self, alpha=0.01, max_iter=1000, bias=False, tol=None, penalty=None, lambda_=0.6 ):
    
        self.alpha = alpha
        self.max_iter = max_iter
        self.bias = bias
        self.tol = tol
        self.penalty = penalty
        self.lambda_ = lambda_

        self.weights = None    
        self.loss_history = []

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

    def _sigmoid(self, z):
        """
        Parameters
        ----------
        z : numpy.ndarray , shape (m_samples, )
            Input values
            
        Returns
        -------
        y_hat : numpy.ndarray , shape (m_samples, )
            Predicted values
        
        """

        return 1 / (1 + np.exp(-z + epsilon))

    def _y_hat(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        w : numpy.ndarray , shape (n_features, )
            Weights

        Returns
        -------
        z : numpy.ndarray , shape (m_samples, )
            Linear fited values before applying sigmoid function 

        """
        return self._sigmoid(np.dot(X, self.weights))

    def _cal_loss(self, y_hat, y_true):
        """
        Calculate the Binary Cross Entropy Loss

        Parameters
        ----------
        y_hat : numpy.ndarray , shape (m_samples, )
            Predicted values

        y_true : numpy.ndarray , shape (m_samples, )
            Target values

        Returns
        -------
        cost : float
            Cost

        """
        # no. of samples
        m = len(y_true)

        # calculate cost in term of y_true and y_hat
        total_cost = np.sum(-y_true * np.log(y_hat + epsilon) - (1 - y_true) * (np.log(1 - y_hat + epsilon))) / m

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
            Gradient of the cost function

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

            y_hat = self._y_hat(X)
            gradient = self._cal_gradient(y_hat, y, X)

            # update the weights
            self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            y_hat = self._y_hat(X)
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

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted Probabilities

        """

        # preprocess the input data
        X = self._preprocess_input_data(X)

        # include the bias term or not
        X = self._bias(X)

        # calculate y_pred
        return self._y_hat(X)

    def predict(self, X, threshold=0.5):

        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        threshold : float
            Threshold value

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        """

        self.threshold = threshold

        y_pred = self.predict_proba(X)

        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0

        return y_pred

    def accuracy(self,X, y_true):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        y_true : numpy.ndarray , shape (m_samples, )
            True values of y

        Returns
        -------
        accuracy : float
            Accuracy

        """
        pred = self.predict(X)
        return np.sum(y_true == pred)/len(y_true)


class OneVsAll(BinaryClassifier):
    def __init__(self, alpha, max_iter, bias=False, tol=None,  penalty=None, lambda_=0.1):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)
        self.penalty = penalty
        self.lambda_ = lambda_


    def fit(self, X, y):

        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
            
        y : numpy.ndarray , shape (m_samples, )
        Target values
        
        Returns
        -------
        None, but updates the weights and loss history attributes
        """

        self.classes = np.unique(y)

        if len(self.classes) < 3:
            super().fit(X, y)
        else:
            if self.bias:
                self.all_weights = np.zeros((len(self.classes), X.shape[1]+1))
            else:
                self.all_weights = np.zeros((len(self.classes), X.shape[1]))

            for i in range(len(self.classes)):
                y_new = np.where(y == self.classes[i], 1, 0)
                super().fit(X, y_new)
                self.all_weights[i] = self.weights

    def predict_proba(self, X):

        if self.weights is None:
            raise AttributeError("You have to fit the model first")


        # X = self._preprocess_input_data(X)
        # X = self._bias(X)

        if hasattr(self, "all_weights"):
            # taken a transpose b/c earlier we were saving our weights in 1-d array
            # but now we are saving the weights in classes * features (classes,features), earlier it was like
            # feature * class (n_features,)
            self.weights = self.all_weights.T
        return super().predict_proba(X)

    def predict(self, X, threshold=0.5):

        if hasattr(self, "all_weights"):
            para_pred = self.predict_proba(X)
            return self.classes[np.argmax(para_pred, axis=1)]
        return super().predict(X, threshold=threshold)


class SoftmaxClassifier(BinaryClassifier):
    def __init__(self, alpha, max_iter, bias=False, tol=None, penalty=None, lambda_=0.6, n_classes=2):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)
        self.n_classes = n_classes

    def _softmax(self, z):
        # stable softmax, to avoid overflow and underflow errors 
        # while calculating softmax for large values
        z -= np.max(z, axis=1, keepdims=True)
        exps = np.exp(z)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _y_hat(self, X):
        z = np.dot(X, self.weights)
        return self._softmax(z)

    def _cal_loss(self, y_hat, y_true):
        m = y_true.shape[0]
        # calculate the loss function
        total_cost = -np.sum(y_true * np.log(y_hat+1e-10))/(2*m)
        # add the regularization term
        if self.penalty is None:
            return total_cost
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sum(np.abs(self.weights))
            return total_cost + regularization
        elif self.penalty == "l2":
            regularization = (self.lambda_ / (2*m)) * np.sum(self.weights[1,:]**2)
            return total_cost + regularization
        else:
            raise ValueError("Invalid penalty type")

    # calculate gradient of the cross entropy loss function
    def _cal_gradient(self, y_hat, y_true, X):
        m = y_true.shape[0]
        grad = np.dot(X.T, y_hat - y_true)/m
        # add the regularization term
        if self.penalty is None:
            return grad
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sign(self.weights)
            gradient = grad + regularization
            gradient[0,:] = grad[0,:]
            return gradient
        elif self.penalty == "l2":
            regularization = (self.lambda_ / m) * self.weights
            gradient = grad + regularization
            gradient[0,:] = grad[0,:]
            return gradient
        else:
            raise ValueError("Invalid penalty type")

    def fit(self, X, y):

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # to include the bias term or not and initialize weights
        X = self._bias(X)


        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(low=-1, high=1, size=(n_features, self.n_classes))
        
        y_encoded = np.zeros((n_samples, self.n_classes))
        for i in range(self.n_classes):
            y_encoded[:, i] = (y == self.classes[i]).astype(int)

        for i in range(self.max_iter):
            y_hat = self._y_hat(X)
            gradient = self._cal_gradient(y_hat, y_encoded, X)

            self.weights = self.weights - (self.alpha * gradient)

            y_hat = self._y_hat(X)
            self.loss_history.append(self._cal_loss(y_hat, y_encoded))


            # Break the loop if loss is not changing much
            if i > 1 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(f"Loss is nan at iteration {i}. Hence, stopping the training")
                break

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        """

        if self.weights is None:
            raise AttributeError("You have to fit the model first")

        # preprocess the input data
        X = self._preprocess_input_data(X)

        # include the bias term or not
        X = self._bias(X)
        
        y_hat = self._y_hat(X)
        return y_hat

    def predict(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        """

        y_hat = self.predict_proba(X)
        return np.argmax(y_hat, axis=1)