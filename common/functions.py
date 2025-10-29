import numpy as np

#损失函数

#MSE、L2
def mean_squared_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)

#交叉熵误差
def cross_entropy(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.size == y_true.size:
        y_pred = np.argmax(y_pred, axis=1)
    n = y_true.shape[0]
    return -np.sum(np.log(y_true[np.arange(n), y_pred] + 1e-10)) / n

#阶跃函数
def step_function0(x):
    return 1 if x > 0 else 0

import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=int)

#sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#TanH函数
def tanh(x):
    return np.tanh(x)

#Relu函数
def relu(x):
    return np.maximum(0, x)

#Leaky Relu函数
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

#Parametric ReLU函数
def parametric_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

#RRelu函数
def rrelu(x, lower=0.01, upper=0.03):
    alpha = np.random.uniform(lower, upper, size=x.shape)
    return np.where(x > 0, x, alpha * x)

#ELU函数
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

#Swish函数
def swish(x):
    return x * sigmoid(x)

#Softplus函数
def softplus(x):
    return np.log1p(np.exp(x))

#Softmax函数
def softmax(x):
    #如果是二维数组，按行计算
    if x.ndim == 2:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

#Identify函数
def identity_function(x):
    return x



if __name__ == '__main__':
    x = np.array([-5.0, 5.0, 0.1])
    print(step_function(x))
    print(sigmoid(x))
    print(tanh(x))
    print(relu(x))
    print(leaky_relu(x))
    print(softmax(x))
    x2 = np.array([[1,2,3],[4,5,6]])
    print(softmax(x2))