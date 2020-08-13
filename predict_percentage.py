# importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pwd

"Collecting data"
df=pd.read_csv('student.csv')
#reading first 5 rows of data
df.head()

#getting the dimensions of data 
df.shape

#getting the insight of data frame
df.describe()

# defining axis
x=df['Hours']
y=df['Scores']

"Plotting scatter plot"
plt.scatter(x,y,label='Data Points',color='red',marker='*',s=25)
plt.title('Predicting Marks')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.grid()
plt.rcParams['axes.facecolor']='yellow'
plt.show()

"finding corelation between the hours and scores"
sns.heatmap(df.corr(),annot=True)

#Dividing data into train and test 
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

"Checking the shape of test and train data set"
x_train.shape

x_test.shape

y_train.shape

y_test.shape

"Training The Algorithm"

#fitting data in the model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

#plotting The Linear Regression Line
line=model.coef_*x + model.intercept_

plt.rcParams['axes.facecolor']= 'yellow'
plt.scatter(x,y,color='red')
plt.plot(x,line,color='black')
plt.show()
model.coef_,model.intercept_

"Data Testing"
#Predecting the result
y_pred=model.predict(x_test)
y_pred

#Actual vs Predicted
pred_data=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
pred_data

#Accuracy Of test dataset
model.score(x_test,y_test)

#Accuracy Of train dataset
model.score(x_train,y_train)

#Visualizating Actual Vs Predicted Comparison Using Barplot
pred_data.plot(kind='bar')
plt.title('Actual VS Predicted',fontsize=25)
plt.xlabel('Hours',fontsize=20)
plt.ylabel('Scores',fontsize=20)
plt.plot()

"Calculating mean absolute and mean squared error"
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print('mean absolute error: ',mean_absolute_error(y_test,y_pred))
print('mean squared error: ',mean_squared_error(y_test,y_pred))

"Predicting The Score Of student Studying for 9.25 hours"
#predicting the marks using the Linear Model
hrs=9.25
prediction=model.predict([[9.25]])
print('The score of student studying 9.25 hours in a day:',prediction)

