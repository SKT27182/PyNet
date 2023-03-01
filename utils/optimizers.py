import numpy as np

class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, dw):
        return self.lr * dw


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta1 = beta
        self.v = None

    def update(self, dw):
        if self.v is None:
            self.v = np.zeros_like(dw)
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        return self.v
    

class RMSprop:
    def __init__(self, lr=0.01, beta=0.99, epsilon=1e-8):
        self.lr = lr
        self.beta2 = beta
        self.epsilon = epsilon
        self.cache = None

    def update(self, dw):
        if self.cache is None:
            self.cache = np.zeros_like(dw)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw ** 2)
        return self.lr * dw / (np.sqrt(self.v) + self.epsilon)


    
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def update(self, dw):
        if self.m is None:
            self.m = np.zeros_like(dw)
            self.v = np.zeros_like(dw)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)