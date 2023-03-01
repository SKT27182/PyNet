import numpy as np


"""
Classes of different activation functions.

- shape of z is (n_[l], batch_size)

- activation functions:
    Returns the activation of z.  (n_[l], batch_size)

- activation_prime functions:
    Returns the derivative of the activation function.  (n_[l], batch_size)

"""

class Tanh:
    def activation(z):
        return np.tanh(z)

    def activation_prime(z):
        return 1.0 - np.tanh(z) ** 2

class ReLU:
    def activation(z):
        return np.maximum(0, z)

    def activation_prime(z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

class Linear:
    def activation(z):
        return z

    def activation_prime(z):
        return np.ones(z.shape)

class Sigmoid:
    def activation(z):
        return 1 / (1 + np.exp(-z))

    def activation_prime(z):
        sigmoid_z = Sigmoid.activation(z)
        return sigmoid_z * (1 - sigmoid_z)

class HardSigmoid:
    def activation(z):
        """
        Hard sigmoid activation function, is used where speed of 
        computation is more important than precision.
        
        Returns
        -------
                0               if z < -2.5, 
                1               if z > 2.5, 
                0.2 * z + 0.5   otherwise.
        """
        return np.maximum(0, np.minimum(1, 0.2 * z + 0.5))

    def activation_prime(z):
        z = HardSigmoid.activation(z)
        return 0.2 * z * (1 - z)

class Softmax:
    def activation(z):
        z -= np.max(
            z, axis=0, keepdims=True
        )  # axis=0 means coloumn z is the shape of (n_l, batch_size), axis=0 means the max value of each column b/c we are giving input as (n_l, batch_size)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def activation_prime(z):
        # we have calculated the dA/dz in the loss_prime itself,
        # that returns (dJ/dA)*(dA/dz) itself so no need to take the derivative of activation here
        return 1


    def softmax_derivative(z):
        softmax_ = Softmax.softmax(z)
        return np.einsum('ij,ik->ij', softmax_, (np.eye(z.shape[0]) - softmax_))
