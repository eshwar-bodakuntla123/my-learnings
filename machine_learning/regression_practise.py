import os
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
parent_dir = os.path.dirname(base_dir)
print(parent_dir)
file_path  = os.path.join(parent_dir,"machine_learning","weather.csv")
dataset = pd.read_csv(file_path)
print(dataset)

x  = dataset['MaxTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

X_train , X_test , Y_train , Y_test = train_test_split(x,y,random_state=0) 