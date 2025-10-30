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

    y_train = y_train.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test