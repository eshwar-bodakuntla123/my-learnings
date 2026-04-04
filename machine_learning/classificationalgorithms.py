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

# plt.show()


'''
Plot the Histogram 
Histogram basically shows the frequency of occurence
it is important because you can understand the correlation bw variables by seeing the graph
'''
import pylab as pl
fruits.drop('fruit_label',axis=1).hist(bins=30,figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('fruit_hist')
# plt.show()


'''
Dividing the data into target and predictor variables
'''
feature_names = ['mass','width','height','color_score',]
X= fruits[feature_names]
y = fruits['fruit_label']

'''Splliting training and testing data'''
from sklearn.model_selection import train_test_split
X_train , X_test,y_train, y_test = train_test_split(X,y,random_state=0)



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
X_test = scalar.transform(X_test)



#Logestic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

print(f'Accuracy of logestic regression classifier on training set : (:.2f){(logreg.score(X_train,y_train))}')
print(f'Accuracy of Logistic regression classifier on test ser (:.2f){logreg.score(X_test,y_test)}')


'''Decision TREE
decision tree are very good with the training dataset because of the process known as 
overfitting - (overfitting occurs whenever your model learns the training data along with all disturbance in training data)
but when it comes to classifing outcome on the testing dataset the accuracy reduces
This is good compared to logestic regression
'''
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
print(f'Accuracy of Decision Tree classifier on trainind set : {clf.score(X_train,y_train)}')
print(f'Accuracy of Decision Tree classifier on test set : {clf.score(X_test,y_test)}')


'''
KNN Classifier
it avoids overfitting
all these value depends the value of K chosen in K-NN , so we need to use elbow method to choose the K value 
in the K nearest neighbour

'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print(f"Accuracy of K-NN classifier on training set : {knn.score(X_train,y_train)}")
print(f"Accuracy of K-NN classifier on test set : {knn.score(X_test,y_test)}")


'''
NavieBayes Classifier
Gaussian is a type of navis bayes classifier

'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
print(f"Accuracy of training data set in GNB classifer {gnb.score(X_train,y_train)}")
print(f"Accuracy of GNB classifier for testing datast : {gnb.score(X_test,y_test)}")

'''
Support vector machine, It is very good on large data set
'''
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
print(f"Accuracy of SVM classifier on traiing set : {svm.score(X_train,y_train)}")
print(f"Accuracy of SVM classifier on testing dataset : {svm.score(X_test,y_test)}")


'''
Result : 

After observing the print results we can see KNN classifier  classified our data set more accurately
we"ll look at the predections that the KNN classifier mean .

'''
#Confusion matrix for knn classifier 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)

'''
Confusion matrix - to show accuracy of KNN algorithm we are using confusion metrix.
confusion matrix A confusion matrix is a table used to evaluate how well a classification model performs
 by comparing actual values vs predicted values.
Actual \ Predicted	Positive	Negative
Positive	True Positive (TP)	False Negative (FN)
Negative	False Positive (FP)	True Negative (TN)

Actual \ Predicted  	Spam	Not Spam
Spam	                 50      	10
Not Spam	              5	        100

Here:

TP = 50
FN = 10
FP = 5
TN = 100

Accuracy=(TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall(Sensitivity) = TP/(TP+FN)
F1 Score =(2*precision*recall)/(precision+recall)
Why Confusion Matrix Matters

Accuracy alone can mislead you.
Example: If 95% emails are not spam, a model predicting always not spam gives 95% accuracy but is useless.

Confusion matrix shows where the model is making mistakes.
'''

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

