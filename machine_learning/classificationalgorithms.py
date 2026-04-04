#Implementing multiple classification algorithms for the same set of problems
import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir,"fruit_data_with_colours.csv")
fruits = pd.read_csv(file_path)
print(fruits.head())
print(fruits.fruit_name.unique())
print(fruits.groupby('fruit_name').size())

