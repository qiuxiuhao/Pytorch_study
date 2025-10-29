#数值微分求导
import numpy as np

#数值微分求导
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(h*2)

#求梯度，x是一个向量
def  _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_grad = x[idx]
        x[idx] = tmp_grad + h
        fxh1 = f(x)
        x[idx] = tmp_grad -h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_grad
    return grad

#传入自变量是一个矩阵
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        for i,x in enumerate(X):
            grad[i] = _numerical_gradient(f, x)
        return grad
