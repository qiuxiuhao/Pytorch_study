import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_gradient

def gradient_descent(f,init_x,lr = 0.01 ,num_iter = 100):
    x = init_x
    #保存x的变化
    x_history = []
    #循环迭代
    for i in range(num_iter):
        x_history.append(x.copy())
        #计算梯度
        grad = numerical_gradient(f,x)
        #更新参数
        x -=lr*grad

    return x ,np.array(x_history)

#定义函数
def f(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    init_x = np.array([-3.0,+4.0])

    lr = 0.05
    num_iter = 50
    #使用梯度下降法，计算最小值点
    x,x_history = gradient_descent(f,init_x,lr = lr,num_iter = num_iter)

    plt.plot([-5,-5],[0,0],'--b')
    plt.plot( [0, 0],[-5, -5], '--b')
    plt.scatter(x_history[:, 0], x_history[:, 1])
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    #plt.legend()
    plt.grid(True)
    plt.show()