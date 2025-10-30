#from nn_base.digit_recognizer import network
from  two_layer_net import TwoLayerNet
from  common.load_data import get_data
import  numpy as np
import  matplotlib.pyplot as plt

"""
二层神经网络，但耗时过于复杂，由于计算梯度采用的数值微分相关方法
预计1个epoch需要1~2h
"""

#1.加载数据
x_train, x_test, t_train, t_test = get_data()

#2.创建model
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#3.设置超参数
learning_rate = 0.1
batch_size = 100
num_epochs = 10

train_size = x_train.shape[0]
iter_per_epoch = np.ceil(train_size / batch_size)
iter_num = int(iter_per_epoch * num_epochs)

train_loss_list = []
train_acc_list = []
test_acc_list = []

#4.循环迭代
for i in range(iter_num):

    #4.1选取数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #4.2计算梯度
    grap  = network.num_gradient(x_batch, t_batch)
    print("grad:=====",i)

    #4.3更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grap[key]

    #4.4计算并保存当前训练损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #4.5每完成一个epoch，计算准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('Epoch:{}, train_acc:{}, test_acc:{}'.format(i/iter_per_epoch, train_acc, test_acc))

#5.画图
x = np.array(len(train_loss_list))
plt.plot(train_acc_list, label='train_acc')
plt.plot(test_acc_list, label='test_acc', linestyle='--')
plt.legend(loc='lower right')
plt.legend(loc='best')
plt.show()
