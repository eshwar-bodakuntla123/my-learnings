import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
file_path = os.path.join(base_dir,"weather.csv")
dataset = pd.read_csv(file_path)
# print(dataset)
# print(dataset.shape)
# print(dataset.describe())
dataset.plot(x='MinTemp',y='MaxTemp',style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.displot(dataset['MaxTemp'])
plt.show()

#Data splicing
x = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)




regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

#To retrive the intercept
print('Intercept', regressor.intercept_)

print('Cofficient', regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
print(df)

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':',linewidth='0.5',color='black')
plt.show()

plt.scatter(X_test,y_test,  color='gray')
plt.plot(X_test,y_pred, color='red',linewidth=2)
plt.show()

print(f'Mean Absolute Error : {metrics.mean_absolute_error(y_test,y_pred)}')
print(f'Mean Squared Error: {metrics.mean_squared_error(y_test,y_pred)}')
print(f'Root Mean Squared Error : {np.sqrt(metrics.mean_squared_error(y_test,y_pred))}')