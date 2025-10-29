import numpy as np
from common.functions import  softmax,cross_entropy
from common.gradient import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    #前向传播
    def forward(self,X):
        a = X @ self.W
        return softmax(a)

    #计算损失值
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy(y, t)
        return loss

#主流程
if __name__ == "__main__":
    #1.定义数据
    x = np.array([0.6,0.9])
    t = np.array([0,0,1])
    #2.定义网络
    net = SimpleNet()
    #3.计算梯度
    f  = lambda _:net.loss(x, t)
    gradw  = numerical_gradient( f  ,net.W)

    print(gradw)
