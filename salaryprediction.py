# importing numpy liberary
import numpy as np

class Linear_Regression():

  #initiating the parameters

  def __init__(self,learning_rate,no_of_iteration):

    self.learning_rate=learning_rate
    self.no_of_iteration=no_of_iteration

  def fit(self,x,y):

    # number of training examples & number of features
    self.m,self.n=x.shape #number of rows and columns

    #initiating the weight(slope) and bias(intercept)

    self.w=np.zeros(self.n) # weight in matrix form
    self.b=0 #bias
    self.x=x # x axis
    self.y=y #y axis

    #implementing the gradient descent algorithm
    for i in range(self.no_of_iteration):
      self.update_weights()

  def update_weights(self,):
    y_predection=self.predict(self.x)

    dw= -(2* (self.x.T).dot(self.y-y_predection))/self.m

    db= -2* np.sum(self.y-y_predection)/self.m

    #updating the weights
    self.w=self.w-self.learning_rate *dw

    self.b=self.b-self.learning_rate*db



  def predict(self,x):

    return x.dot(self.w)+self.b

"""# Using regression model to prediction"""

#importing the pandas library
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""**Data preprocessing**"""

#loading the data set
salary_data=pd.read_csv("/content/salary_data (3).csv")
salary_data.head()

salary_data.tail()

salary_data.shape

#checking for null values in data set
salary_data.isnull().sum()

x=salary_data.iloc[:,:-1].values #assigning the value on x-axis
y=salary_data.iloc[:,1].values #assigning the value on y-axis

print(x)

print(y)



"""spliting the data set test and training"""

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=2) #training the model using train_test_split() function



"""Training the model"""

model=Linear_Regression(learning_rate=0.02,no_of_iteration=1000)

model.fit(x_train,y_train)

#printing the parameters valuse (weight & bias)

print('model = ',model.w[0])
print('bias = ',model.b)

"""predict the salary for test data

"""

test_data_prediction=model.predict(x_test)

print(x_test)
print(test_data_prediction)

"""Visualizing the predicted value and actual value"""

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,test_data_prediction,color='blue')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.title('Salary VS Experience')
plt.show()