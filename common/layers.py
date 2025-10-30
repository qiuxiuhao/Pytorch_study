import numpy as np


#Relu
class Relu:

    #初始化
    def __init__(self):
        #记录哪些X<=0
        self.mask =None

    def forward(self,x):
        self.mask = (x <= 0)
        y = x.copy()
        # 将x<=0的值都赋值为0
        y[self.mask] = 0
        return y

    #反向传播
    def backward(self,dy):
        dx = dy.copy()
        dx[self.mask] = 0
        return dx

#Sigmoid
class Sigmoid:

    def __init__(self):
        self.y = None

    def forward(self,x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self,dy):
        return  dy * self.y * ( 1.0 - self.y)

