import  pandas as pd
import  numpy as np
import  joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from  common.functions import sigmoid,softmax

#数据来源：https://www.kaggle.com/competitions/digit-recognizer/data
#当前代码不完整，缺少模型参数文件

#读取数据
def get_data():
    #从文件加载数据集
    data = pd.read_csv('../data/train.csv')
    #划分数据集
    x = data.drop('label', axis=1)
    y = data['label']

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    #特征工程、归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test, y_test

x,y = get_data()

#1.初始化神经网络
def init_network():

    network = joblib.load('')
    return network

#2加载参数
network = init_network()

#3.前向传播
batch_size = 100
accuracy_cnt = 0
n = x.shape[0]

for i in range(0,n,batch_size):
    #3.1取数据
    x_batch = x[i:i+batch_size]

    #3.2前向传播
    y_batch = forward(network,x_batch)

    y_pred = np.argmax(y_batch,axis=1)

    #3.4累加准确个数
    accuracy_cnt += np.sum((y_pred==y[i:i+batch_size]))

