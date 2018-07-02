#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
X = dataset.iloc[:, 0:26].values
y = dataset.iloc[:,26].values

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing the deep learning libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 26))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 12, kernel_initializer = 'uniform', activation = 'relu'))

#adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

#predicting the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


