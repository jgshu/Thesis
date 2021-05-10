import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# from lstm_model import RNN_LoadForecastser
from lstm_model_2 import LoadForecastser
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# 超参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 导入要inverse transform的df数据,生成userID和coName的键值对
csv_path = './data/type_10_csv_after_anomaly_detection/'
file_list = os.listdir(csv_path)
user_id_df_dict = {}
user_id_co_name_dict = {}
for i in range(len(file_list)):
    file = file_list[i]
    coName, userID = file[:-4].split('_')
    print(coName)
    df = pd.read_csv(csv_path + file)
    user_id_df_dict[userID] = df
    user_id_co_name_dict[userID] = coName

user_id_list = list(user_id_co_name_dict.keys())
k = 0
userID = user_id_list[k]
coName = user_id_co_name_dict[userID]
print(coName)
user_df = user_id_df_dict[userID].copy()
col_load = user_df.loc[:, 'load'].values.reshape(-1, 1)

# 提取load的最大最小值
scaler = MinMaxScaler()
scaler = scaler.fit(col_load)
del user_df
del col_load
del user_id_df_dict

# 导入训练集和测试集
npy_path = './output/train_test/type10/%s_%s/' % (coName, userID)
# train_x = np.load(path + 'train_x.npy')
# train_y = np.load(path + 'train_y.npy')
test_x = np.load(npy_path + 'test_x.npy')
test_y = np.load(npy_path + 'test_y.npy')


# numpy to tensor
test_x = torch.from_numpy(test_x).type(torch.Tensor)
test_y = torch.from_numpy(test_y).type(torch.Tensor)

# model = torch.load('./model.pkl')

model.eval()

# 模型用于测试集
test_x = test_x.to(device)
test_y_pred = model(test_x).to(device)

test_y = scaler.inverse_transform(test_y)
print(test_y.shape)

test_y_pred = scaler.inverse_transform(test_y_pred.detach().cpu().numpy())
print(test_y_pred.shape)

testScore = math.sqrt(mean_squared_error(test_y[:, 0], test_y_pred[:, 0]))
print('Test Score: %.2f RMSE' % testScore)

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 30,
       }

fig, ax = plt.subplots(figsize=(20, 10))

y1 = test_y[0 : 96 : , :].reshape(7 * 96)
y2 = test_y_pred[-7 * 5:-7 * 4, :].reshape(7 * 96)
x = range(0, 7 * 96)
ax.plot(x, y1)
ax.plot(x, y2)
plt.xticks(font=font)
plt.yticks(font=font)
ax.set_ylabel('7 days', font=font)
ax.set_xticks(np.arange(0, 7 * 96, 96))
ax.set_xticklabels(np.arange(0, 7))
plt.show()
