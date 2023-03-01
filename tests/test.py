#!/home/shailja/.virtualenv/my_env/bin/python

# module for testing
from sklearn import datasets
from ml.linear_reg import GradientDescent as gd
from ml.logistic_reg import Classifier as cl
import sys


class MLModels:
    def __init__(self, type = "logistic", optimizer = "batch", alpha = 0.01, max_iter = 1000, tol = 1e-4, bias = True, penalty = None, lambda_ = 0.01):
        self.type = type
        self.optimizer = optimizer
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.bias = bias
        self.penalty = penalty
        self.lambda_ = lambda_


    def load_data(self):
        if self.type == "linear":
            data =  datasets.load_diabetes()
            x_data = data.data
            y_data = data.target
        elif self.type == "binary":
            data = datasets.load_breast_cancer()
            x_data = data.data
            y_data = data.target
        elif self.type == "multi":
            data = datasets.load_iris()
            x_data = data.data
            y_data = data.target

        return x_data, y_data
    
    def train(self, x_data, y_data):
        if self.type == "linear":
            if self.optimizer == "batch":
                model = gd.BatchGD(alpha =self.alpha, max_iter= self.max_iter, bias = self.bias, tol = self.tol, penalty = self.penalty, lambda_ = self.lambda_)
                model.fit(x_data, y_data)
                return model
            
            elif self.optimizer == "minibatch":
                model = gd.MiniBatchGD(alpha =self.alpha, max_iter= self.max_iter, bias = self.bias, tol = self.tol, penalty = self.penalty, lambda_ = self.lambda_)
                model.fit(x_data, y_data, batch_size = 64)
                return model
            
            elif self.optimizer == "stochastic":
                model = gd.StochasticGD(alpha =0.001, max_iter= 5000, bias = self.bias, tol = self.tol, penalty = self.penalty, lambda_ = self.lambda_)
                model.fit(x_data, y_data)
                return model
            
            elif self.optimizer == "linear_search":
                model = gd.LinearSearchGD( max_iter= self.max_iter, bias = self.bias, tol = self.tol, penalty = self.penalty, lambda_ = self.lambda_)
                model.fit(x_data, y_data)
                return model
            
            else:
                print("Optimizer not found")
                return None
        elif self.type == "binary":
            if self.optimizer == "batch":
                model = cl.BinaryClassifier(alpha =self.alpha, max_iter= self.max_iter, bias = self.bias, tol = self.tol, penalty = self.penalty, lambda_ = self.lambda_)
                model.fit(x_data, y_data)
                return model
        elif self.type == "multi":
            if self.optimizer == "ova":
                model = cl.OneVsAll(alpha =self.alpha, max_iter= self.max_iter, bias = self.bias, tol = self.tol, penalty = self.penalty, lambda_ = self.lambda_)
                model.fit(x_data, y_data)
                return model
            elif self.optimizer == "softmax":
                model = cl.SoftmaxClassifier(alpha =self.alpha, max_iter= self.max_iter, bias = self.bias, tol = self.tol, penalty = self.penalty, lambda_ = self.lambda_)
                model.fit(x_data, y_data)
                return model
            

def main():
    type = sys.argv[1]
    optimizer = sys.argv[2]
    # if thrird argument is not given then it will take default value
    if len(sys.argv) == 4:
        max_iter = int(sys.argv[3])
    else:
        max_iter = 10000
    model = MLModels(type=type, optimizer=optimizer, alpha=0.01, max_iter=max_iter, tol=1e-10, bias=True, penalty="l2", lambda_=0.01)
    x_data, y_data = model.load_data()
    model = model.train(x_data, y_data)
    if type == "linear":
        print(model.r2_score(x_data, y_data))

    elif type == "binary" or type == "multi":
        print(model.accuracy(x_data, y_data))

if __name__ == "__main__":
    main()


"""
 ./test.py {type:[linear, binary, multi]} {optimizer:[batch, minibatch, stochastic, linear_search, ova, softmax]} {max_iter}

"""