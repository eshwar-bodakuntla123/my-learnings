# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# data = pd.read_csv('')

# data= data.drop()

# #Dimensions of dataset
# n = data.shape[0]
# p = data.shape[1]

# #make data a numpy  array
# data =data.values

# #training and test data
# train_start = 0
# train_end = int(np.floor(0.8*n))
# test_start - train_end
# test_end = n

# data_train  = data[np.arange(train_start,train_end),:]
# data_test = data[np.arange(data_test,test_end),:]

# #scale data

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data_train = scaler.fit_transform(data_train)
# data_test = scaler.transform(data_train)

 
s1 = 'GeeksforGeeks'

s2 = lambda func: func.upper()
print(s2(s1))