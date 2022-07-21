#############################################################################################################

# Dataset : https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

############################################ Importing Libraries ############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings


############################################# Importing Dataset #############################################

datset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
datset.columns=['Sepal_Len','Sepal_Wid','Petal_Len','Petal_Wid','Iris_Class']
datset.head()
datset.describe()
datset.info()


###################################### Checking the Variation of Data ######################################

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(datset['Sepal_Len'])
plt.show()
# datset['Sepal_Len'].hist()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(datset['Sepal_Wid'])
plt.show()
# datset['Sepal_Wid'].hist()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(datset['Petal_Len'])
plt.show()
# datset['Petal_Len'].hist()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(datset['Petal_Wid'])
plt.show()
# datset['Petal_Wid'].hist()


################################### Finding best parameters having relation ##################################

%matplotlib inline
sns.set()
sns.pairplot(datset, hue='Iris_Class', height=3)


################################# Classifying the Data using best parameters #################################

colors = ['blue', 'red', 'green']
Iris_Class = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = datset[datset['Iris_Class'] == Iris_Class[i]]
    plt.scatter(x['Petal_Len'], x['Petal_Wid'], c = colors[i], label=Iris_Class[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


################################### Assigning Labels for better distinction ##################################

from sklearn.preprocessing import LabelEncoder
Label_Enc = LabelEncoder()
datset['Iris_Class'] = Label_Enc.fit_transform(datset['Iris_Class'])
datset.head()


############################################# Training the model #############################################

from sklearn.model_selection import train_test_split
x_axis = datset.drop(columns=['Iris_Class'])
y_axis = datset['Iris_Class']
x_train, x_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=0.2)
# 80% Train, 20% Test


############################################# Testing the model #############################################

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
print("Accuracy: ",model.score(x_test, y_test) * 100)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
print("Accuracy: ",model.score(x_test, y_test) * 100)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)
print("Accuracy: ",model.score(x_test, y_test) * 100)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Accuracy: ",model.score(x_test, y_test) * 100)


################################################### The End ###################################################
