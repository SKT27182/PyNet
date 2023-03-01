import numpy as np
from dl.nn.activations import *
from utils import optimizers
from dl.nn.helper import *
from copy import deepcopy

class Layer:

    """
    Base class for all the layers in the neural network.
    """

    def __init__(self):
        """
        Initialize the input and output for the layer.
        """
        self.input = None
        self.output = None
        self.input_shape = 0
        self.units = 0
        self.initializer = None
        self.optimizer_w = optimizers.Adam()
        self.optimizer_b = optimizers.Adam()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}'


    def forward_propagation(self, input):

        """
        Implement forward propagation through the layer.

        Parameters
        ----------
        input: numpy.ndarray, shape (n_[l-1], batch_size)
            Input data to be propagated through the layer.

        Returns
        -------
        output: numpy.ndarray, shape (n_[l], batch_size)
            Output of the layer after forward propagation.

        """

        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):

        """
        Implement backward propagation through the layer.

        Parameters
        ----------

        output_error: numpy.ndarray
            Error in the output of the layer.

        learning_rate: float
            Learning rate to be used for weight updates.

        Returns
        -------
        numpy.ndarray
            Error to be propagated back to the previous layer.
        """

        raise NotImplementedError
    
    
class BatchNormalization(Layer):
    
        """
        Class for batch normalization layer in the neural network.
    
        Parameters
        ----------
    
        n_l: int
            Number of neurons in the output layer.
        """
    
        def __init__(self, units):
    
            """
            Initialize the batch normalization layer with weights and biases.
    
            Parameters
            ----------
            n_l: int
                Number of neurons in the output layer.
            """
    
            super().__init__()
            self.gamma = np.ones((units, 1))  # gamma (n_[l], 1)
            self.beta = np.zeros((units, 1))  # beta  (n_[l], 1)
            self.epsilon = 1e-8
            self.units = units
    
        def forward_propagation(self, activate_l_1):
    
            """
            Implement forward propagation through the batch normalization layer.
    
            Parameters
            ----------
            activate_l_1: numpy.ndarray
                activate_l_1 data to be propagated through the layer.
    
            Returns
            -------
            numpy.ndarray
                z of the layer after forward propagation.
            """
    
            self.input = activate_l_1  # activate (n_[l], batch_size)
            self.mean = np.mean(self.input, axis=1, keepdims=True)  # mean (n_[l], 1)
            self.variance = np.var(self.input, axis=1, keepdims=True)  # variance (n_[l], 1)
            self.std = np.sqrt(self.variance + self.epsilon)  # std (n_[l], 1)
            self.z = (self.input - self.mean) / self.std  # z (n_[l], batch_size)
            self.output = self.gamma * self.z + self.beta  # output (n_[l], batch_size)
    
            return self.output  # output (n_[l], batch_size)
    
        def backward_propagation(self, output_error):
    
            """
            Implement backward propagation through the batch normalization layer.
    
            Parameters
            ----------
            output_error: numpy.ndarray
                Error in the output of the layer.
    
            Returns
            -------
            numpy.ndarray
                Error to be propagated back to the previous layer.
            """

            self.dgamma = np.sum(output_error * self.z, axis=1, keepdims=True)
            self.dbeta = np.sum(output_error, axis=1, keepdims=True)
            self.dz = output_error * self.gamma
            self.dvariance = np.sum(self.dz * (self.input - self.mean) * (-0.5) * (self.variance + self.epsilon) ** (-1.5), axis=1, keepdims=True)
            self.dmean = np.sum(self.dz * (-1) / self.std, axis=1, keepdims=True) + self.dvariance * np.sum(-2 * (self.input - self.mean), axis=1, keepdims=True) / self.input.shape[1]
            self.dinput = self.dz / self.std + self.dvariance * 2 * (self.input - self.mean) / self.input.shape[1] + self.dmean / self.input.shape[1]

            self.gamma -= self.optimizer_w.update(self.dgamma)
            self.beta -= self.optimizer_b.update(self.dbeta)

            return self.dinput

class DenseLayer(Layer):

    """
    Class for fully connected layer in the neural network.

    Parameters
    ----------

    n_l_1: int
        Number of neurons in the input layer.

    n_l: int
        Number of neurons in the output layer.
    """

    def __init__(self, input_shape=None,units=None, activation=None, l1=0, l2=0, use_bias=True, initializer="glorot"):

        """
        Initialize the fully connected layer with weights and biases.

        Parameters
        ----------
        n_l_1: int
            Number of neurons in the input layer.

        n_l: int
            Number of neurons in the output layer.

        activation: str, optional
            Activation function to be used in the layer.
            for adding user defined activation function. use the ActivationLayer .

        l1: float, optional
            L1 regularization parameter.

        l2: float, optional
            L2 regularization parameter.

        use_bias: bool, optional
            Whether to use bias in the layer or not.

        initializer: str, optional
            Weight initializer to be used for initializing the weights.
        """

        super().__init__()

        if input_shape is not None:
            self.input_shape = input_shape    # (n_[l-1], batch_size)

        # set the number of neurons in this layer
        self.units = units             # n_[l]

        # whether to use bias in the layer or not
        self.use_bias = use_bias

        # set the regularization parameters
        self.l1 = l1
        self.l2 = l2


        # set the activation function for the layer if specified by the user
        if activation is not None:
            if activation.lower() not in ['sigmoid', 'tanh', 'relu', 'softmax', "linear", "hard_sigmoid"]:
                raise ValueError("Invalid activation function.")
            else:
                if activation.lower() == "linear":
                    self.activation = ActivationLayer(Linear)
                elif activation.lower() == "hard_sigmoid":
                    self.activation = ActivationLayer(HardSigmoid)
                elif activation.lower() == "softmax":
                    self.activation = ActivationLayer(Softmax)
                elif activation.lower() == "sigmoid":
                    self.activation = ActivationLayer(Sigmoid)
                elif activation.lower() == "tanh":
                    self.activation = ActivationLayer(Tanh)
                elif activation.lower() == "relu":
                    self.activation = ActivationLayer(ReLU)

        else:
            self.activation = None

    def initialize(self, optimizer=optimizers.Adam(), initializer="glorot"):
        """
        Initialize the optimizer for the layer.

        Parameters
        ----------
        optimizer: Optimizer
            Optimizer to be used for updating the weights and biases.

        initializer: str, optional
            Weight initializer to be used for initializing the weights.

        """
        self.optimizer_w = deepcopy(optimizer)

        self.weights = initializers(method=initializer, shape=(self.units, self.input_shape))
        if self.use_bias:
            self.biases = initializers(method=initializer, shape=(self.units, 1))
            self.optimizer_b = deepcopy(optimizer)
        else :
            self.biases = 0
        # include bias parameters only if use_bias is True
        
        self.trainable_params = self.weights.size + self.units * self.use_bias
        


    def forward_propagation(self, activate_l_1):

        """
        Implement forward propagation through the fully connected layer.

        Parameters
        ----------
        activate_l_1: numpy.ndarray
            activate_l_1 data to be propagated through the layer.

        Returns
        -------
        numpy.ndarray
            z of the layer after forward propagation.
        """
            

        self.input = activate_l_1  # activate (n_[l], batch_size)
        z_l = (
            np.dot(self.weights, self.input) + self.biases
        )  # z = weights x input + biases = (n_[l], n_[l-1]) x (n_[l-1], batch_size) + (n_[l], 1) = (n_[l], batch_size)

        # Apply activation function if present
        if self.activation is not None:
            z_l = self.activation.forward_propagation(z_l)

        self.output = z_l

        return self.output  # z (n_[l], batch_size)

    def backward_propagation(self, output_error):

        """
        Implement backward propagation through the fully connected layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            Error in the output of the layer.

        Returns
        -------
        numpy.ndarray
            Error to be propagated back to the previous layer.
        """

        # Apply activation function if present in the dense layer itself
        if self.activation is not None:
            output_error = self.activation.backward_propagation(output_error)

        # calculating the error with respect to weights before updating the weights
        input_error = np.dot(
            self.weights.T, output_error
        )  # weights x output_error  (n_[l], n_[l-1]) x (n_[l], batch_size) = (n_[l-1], batch_size)
        weights_error = np.dot(
            output_error, self.input.T
        )  # output_error x input    (n_l, batch_size) x (n_[l-1], batch_size) = (n_[l], n_[l-1])

        m_samples = output_error.shape[1]

        # addition of regularization term, by default both l1 and l2 are 0
        reg_term = (self.l1 / m_samples) * np.sign(self.weights)
        reg_term = (self.l2 / m_samples) * self.weights


        self.weights -= self.optimizer_w.update( weights_error + reg_term)

        if self.use_bias:
            self.biases -= np.sum(self.optimizer_b.update( output_error), axis=1, keepdims=True)

        return input_error

class ActivationLayer(Layer):

    """
    Activation layer for neural networks.
    This layer applies an non-linearity to the input data.

    Parameters:
    -----------

    activation (class) : (callable)
        The class of the activation function to be used. The class should have two methods:
        activation and activation_prime.
        activation:
            The activation function.

        activation_prime:
            The derivative of the activation function.

    """

    def __init__(self, activation):
        super().__init__()
        self.activation = activation.activation
        self.activation_prime = activation.activation_prime
        self.activation_name = activation.__name__


    def forward_propagation(self, z_l):

        """
        Perform the forward propagation of the activation layer.

        Parameters:
        z (numpy.ndarray): The z to the layer.

        Returns:
        numpy.ndarray: The output of the layer after applying the activation function.
        """

        self.input = z_l
        activate_l = self.activation(self.input)
        self.output = activate_l

        return self.output  # (n_[l], batch_size)

    def backward_propagation(self, output_error):

        """
        Perform the backward propagation of the activation layer.

        Parameters:
        -----------

        output_error (numpy.ndarray):
            The error that needs to be backpropagated through the layer.

        Returns:
        --------

        numpy.ndarray:
            The input error after backward propagation through the activation layer.
        """
        # if self.activation_prime == None:
        #     return output_error

        del_J_del_A_l = np.multiply(
            self.activation_prime(self.input), output_error
        )  # element-wise multiplication (n_[l], batch_size) x (n_[l], batch_size) = (n_[l], batch_size)

        return del_J_del_A_l

class DropoutLayer(Layer):

    """
    Dropout layer for neural networks.
    This layer randomly drops out units during training to prevent overfitting.

    Parameters:
    ----------

    dropout_rate (float):
        The dropout rate. The rate of neurons that will be dropped out.
        It should be a float in the range of [0, 1].

    """

    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.trainable_params = 0
    def forward_propagation(self, input):

        """
        Perform the forward propagation of the dropout layer.

        Parameters:
        -----------

        input (numpy.ndarray):
            The input to the layer.

        Returns:
        --------

        numpy.ndarray:
            The output of the layer after applying dropout.
        """

        self.input = input  # input (n_[l], batch_size)

        # sample from a binomial distribution with p = 1 - dropout_rate
        self.mask = np.random.binomial(
            1, 1 - self.dropout_rate, size=input.shape
        )  # mask (n_[l], batch_size)

        self.output = (self.input * self.mask) / ( 1 - self.dropout_rate)  # (n_[l], batch_size)
        return self.output  # (n_[l], batch_size)

    def backward_propagation(self, output_error):

        """
        Perform the backward propagation of the dropout layer.

        Parameters:
        -----------

        output_error (numpy.ndarray):
            The error that needs to be backpropagated through the layer.

        Returns:
        --------

        numpy.ndarray:
            The input error after backward propagation through the dropout layer.
        """

        return (output_error * self.mask ) / ( 1 - self.dropout_rate)  # (n_[l], batch_size)

class FlattenLayer:
    
        """
        Flatten layer for 2D inputs.
    
        Parameters:
        -----------
        None
    
        Attributes:
        -----------
        None
        """
    
        def forward_propagation(self, inputs):
            """
            Perform forward propagation for the flatten layer.
    
            Parameters:
            -----------
            inputs: numpy.ndarray
                The input data with shape (height, width, channels, samples).
    
            Returns:
            --------
            output: numpy.ndarray
                The output data after flattening, with shape ( height * width * channels, samples).
            """
    
            self.input_height, self.input_width, self.channels, self.samples = inputs.shape
    
            # Flatten the input
            output = inputs.reshape(self.input_height * self.input_width * self.channels, self.samples)
    
            return output
        
        def backward_propagation(self, dL_doutput):
            """
            Perform backward propagation for the flatten layer.
    
            Parameters:
            -----------
            dL_doutput: numpy.ndarray
                The gradient of the loss with respect to the output data.
    
            Returns:
            --------
            dL_dinputs: numpy.ndarray
                The gradient of the loss with respect to the input data.
            """
    
            # Reshape the gradient of the loss with respect to the output data
            dL_dinputs = dL_doutput.reshape(self.input_height, self.input_width, self.channels, self.samples)
    
            return dL_dinputs