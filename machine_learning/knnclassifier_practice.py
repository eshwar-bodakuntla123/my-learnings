import os
import pandas as pd

from sklearn.model_selection import train_test_split

base_dir  = os.path.dirname(os.path.abspath(__file__))


file_path = os.path.join(base_dir,"fruit_data_with_colours.csv")
fruits = pd.read_csv(file_path)
feature_names = ['mass','width','height','color_score',]

X = fruits[feature_names]
y = fruits['fruit_label']


X_train , X_test , y_train, y_test = train_test_split(X,y,random_state=0)

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print(f"Knn Accuracy on training dataset : {knn.score(X_train,y_train)}")
print(f"Knn Accuracy on testing dataset : {knn.score(X_test,y_test)}")
print(fruits.head())

x1_test = {
    'mass':int(input("enter mass")),'width':float(input("enter width")),'height':float(input("enter height"))
    ,'color_score':float(input("enter colour score"))
}
df_input  = pd.DataFrame([x1_test])


df_input_scaled  = scalar.transform(df_input)
output1 = knn.predict(df_input_scaled)

label_to_name = dict(zip(fruits.fruit_label,fruits.fruit_name))

print(f"It is of fruitlabel : {label_to_name[output1[0]]}")
