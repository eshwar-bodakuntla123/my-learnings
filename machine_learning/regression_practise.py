import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("weather.csv")
print(dataset)

x  = dataset['MaxTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

X_train , X_test , Y_train , Y_test = train_test_split(x,y,random_state=0) 