import numpy as np

from  common.functions import softmax,sigmoid,cross_entropy
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self,x):
        w1,w2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']
        a1 = x @ w1 + b1
        z1 = sigmoid(a1)
        a2 = z1 @ w2 + b2
        y  = softmax(a2)
        return y

    def loss(self,x,t):
        y = self.forward(x)
        loss = cross_entropy(y,t)
        return loss

    def accuracy(self,x,t):
        y_pro = self.forward(x)#预测概率
        y = np.argmax(y_pro,axis=1)#取最大概率的分类号
        correct = np.sum(y==t)
        return correct/x.shape[0]

    #计算梯度
    def num_gradient(self,x,t):
        #定义目标函数
        loss_f = lambda _:self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_f,self.params['W1'])
        grads['W2'] = numerical_gradient(loss_f,self.params['W2'])
        grads['b1'] = numerical_gradient(loss_f,self.params['b1'])
        grads['b2'] = numerical_gradient(loss_f,self.params['b2'])
        return grads





