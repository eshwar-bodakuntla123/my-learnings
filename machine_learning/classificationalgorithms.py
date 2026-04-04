#Implementing multiple classification algorithms for the same set of problems
import os
import pandas as pd
import numpy as np
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir,"fruit_data_with_colours.csv")
fruits = pd.read_csv(file_path)
print(fruits.shape)
print(fruits.head())
print(fruits.fruit_name.unique())
print(fruits.groupby('fruit_name').size())


# to display count of each fruit 
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(fruits['fruit_name'],label="count")
# plt.show()


'''
The below one is box plot, box plot mainly used to detect any sort of outliers in the data
box plot spans the IQR (
IQR Meaning-->    
        Sort the data in ascending order.
        Find Q1 (median of the lower half of the data).
        Find Q3 (median of the upper half of the data).
        Calculate IQR = Q3 − Q1, which represents the spread of the middle 50% of the data.
        In a box plot, the box spans from Q1 to Q3, and this distance is the Interquartile Range (IQR).
)
It detects the data points that lie outside the average range
'''
fruits.drop('fruit_label',axis=1).plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False
                                        ,figsize=(9,9),title='Box Plot for each input variable')

plt.show()


'''
Plot the Histogram 
Histogram basically shows the frequency of occurence
it is important because you can understand the correlation bw variables by seeing the graph
'''
import pylab as pl
fruits.drop('fruit_label',axis=1).hist(bins=30,figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('fruit_hist')
plt.show()


'''
Dividing the data into target and predictor variables
'''
feature_names = ['mass','width','height','color_score',]
X= fruits[feature_names]
y = fruits['fruit_label']

'''Splliting training and testing data'''
from sklearn.model_selection import train_test_split
X_train , x_test,Y_train, y_test = train_test_split(X,y,random_state=0)


'''Scaling :
Our raw data can be very biased (inaccurate) , it is very important to normalize your data.
ex : mass ranging in double digit , height in single digit , colour in single digit 
due to this your output data is very biased (inaccurate). 
This is the reason we need to scale the data in such a way that all of these values 
will have a similar range that's exactly what the scalar function does.

'''
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
X_train  = scalar.fit_transform(X_train)
X_test = scalar.transform(x_test)



#Logestic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)

print(f'Accuracy of logestic regression classifier on training set : (:.2f){(logreg.score(X_train,Y_train))}')
print(f'Accuracy of Logistic regression classifier on test ser (:.2f){logreg.score(X_test,y_test)}')

