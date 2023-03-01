import numpy as np
import pandas as pd


class Loss:

    """
    Base class for loss functions.

    Parameters
    ----------
    n_classes : int, optional
        Number of classes in the dataset, by default 1

    Attributes
    ----------
    n_classes : int
        Number of classes in the dataset

    Methods
    -------
    loss(y_true, y_pred)
        Compute the loss function.

    loss_prime(y_true, y_pred)
        Compute the derivative of the loss function with respect to the predicted output.

    """

    def __init__(self, l1 = 0, l2 = 0) -> None:
        self.l1 = l1
        self.l2 = l2

    def loss(self, y_true, y_pred):
        """
        Compute the loss function.

        Parameters
        ----------
        y_true : numpy.ndarray
            True labels of the dataset. shape = ( batch_size,  ), and for classification one-hot encoded (n_classes, batch_size)
            Returns: scalar

        y_pred : numpy.ndarray
            Predicted labels of the dataset. shape = (n_classes, batch_size)
            Returns: (n_classes, batch_size)

        Returns
        -------
        float
            Loss value.

        """
        raise NotImplementedError

    def loss_prime(self, y_true, y_pred):
        """
        compute the loss derivative with respect to the predicted output.

        Parameters
        ----------

        y_true : numpy.ndarray
            True labels of the dataset.

        y_pred : numpy.ndarray
            Predicted labels of the dataset.

        Returns
        -------
        numpy.ndarray, shape (n_classes, batch_size)
            Derivative of the loss function with respect to the predicted output.

        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return self.__class__.__name__
            
class MSE(Loss):

    """
    Mean squared error loss function.
    """

    def loss(self, y_true, y_pred, ws):
        m_samples = y_pred.shape[1]

        cost = np.sum(np.power(y_pred - y_true, 2)) / (2 * m_samples)
            
        reg_term = 0
        reg_term += (self.l1/m_samples) * ws[0]
        reg_term += (self.l2/(2*m_samples)) * ws[1]

        return cost + reg_term

    def loss_prime(self, y_true, y_pred):
        m_samples = y_pred.shape[1]

        cost_prime =  (y_pred - y_true) / (m_samples)

        return cost_prime

class MAE(Loss):

    """
    Mean absolute error loss function.
    """


    def loss(self, y_true, y_pred, ws):
        m_samples = y_pred.shape[1]

        cost = np.sum(np.abs(y_pred - y_true)) / (m_samples)

        reg_term = 0
        reg_term += (self.l1/m_samples) * ws[0]
        reg_term += (self.l2/(2*m_samples)) * ws[1]
        
        return cost + reg_term

    def loss_prime(self, y_true, y_pred):
        m_samples = y_pred.shape[1]
        
        cost_prime = np.sign(y_true - y_pred) / m_samples

        return cost_prime 

class BinaryCrossentropy(Loss):

    """
    Binary cross-entropy loss function.
    """

    def loss(self, y_true, y_pred, ws):
        m_samples = y_pred.shape[1]
        cost = (
            -np.sum(
                y_true * np.log(y_pred + 1e-10)
                + (1 - y_true) * np.log(1 - y_pred + 1e-10)
            )
            / m_samples
        )

        reg_term = 0
        reg_term += (self.l1/m_samples) * ws[0]
        reg_term += (self.l2/(2*m_samples)) * ws[1]
        
        return cost + reg_term

    def loss_prime(self, y_true, y_pred):
        m_samples = y_pred.shape[1]

        cost_prime = ((y_pred - y_true) / (y_pred * (1 - y_pred + 1e-10))) / m_samples

        return cost_prime 
   
class CategoricalCrossentropy(Loss):

    """
    Categorical cross-entropy loss function.
    """

    def loss(self, y_true, y_pred, ws):
        m_samples = y_pred.shape[1]

        cost = -np.sum(y_true * np.log(y_pred + 1e-10))   # shape = (batch_size,)

        cost = cost / m_samples  

        reg_term = 0
        reg_term += (self.l1/m_samples) * ws[0]
        reg_term += (self.l2/(2*m_samples)) * ws[1]

        return cost + reg_term
    
    def loss_prime(self, y_true, y_pred):
        m_samples = y_pred.shape[1]

        # this is little bit different from the else loss_prime,
        # this return the (dJ/dA)*(dA/dz) so we don't need to find the derivative of sofmax_prime 
        cost_prime = (y_pred - y_true) / m_samples

        return cost_prime