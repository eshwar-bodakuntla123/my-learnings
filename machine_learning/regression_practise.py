import os
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

bas_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(bas_dir,"weather.csv")
dataset = df = pd.read_csv(file_path)
print(dataset)

x  = df['MinTemp'].values.reshape( -1,1)
y  = df['MaxTemp'].values.reshape(-1,1)


X_train ,X_test , Y_train , Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

reg = LinearRegression()
reg.fit(X_train,Y_train)
c = reg.coef_
m = reg.intercept_
print(f"y = {m}x+{c}")
y_predict = reg.predict(X_test)
# y  mx + c
resdf = pd.DataFrame()
resdf['y_test'] = Y_test.flatten()
resdf['y_predict'] = y_predict.flatten()
print(resdf)

print(f"mean absoulte error : {metrics.mean_absolute_error(Y_test,y_predict)}")
print(f"Mean square error : {metrics.mean_squared_error(Y_test,y_predict)}")
print(F"ROOT MEAN SQUARE ERROR : {np.sqrt(metrics.mean_squared_error(Y_test,y_predict))}")