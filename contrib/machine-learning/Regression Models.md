# Regression Models
Regression is a technique to examine the relationship between the Dependant variable (Y) and Independent Variables(X).
It is a type of supervised Learning.
There are many types of Regression models:
1. Linear Regression Model
2. Multiple Regression Model
3. Logistic Regression Model
4. Ridge Regression Model.
5. Lasso Regression Model
   
In order to perform Regression, we need to split our data into training and test sets using `train_test_split` model from `sklearn.model_selection` library.

Here is the explanation to create the training and test set from the dataset.

1. Here we are importing the relevant libraries and the models. In the sklearn library , we are already provided some datasets like - diabetes, breast_cancer,iris etc. In the following code we have laoded the diabetes dataset. 
   `````
   import pandas as pd
   import numpy as np
   from sklearn.datasets import load_diabetes
   from sklearn.model_selection import train_test_split
   `````
2. Converting the dataset into the Pandas Dataframe.
   `````
   df=load_diabetes()
   dataset=pd.DataFrame(df.data)
   dataset.columns=df.feature_names
   X=dataset
   Y=df.target
   `````
3. Spliting the Data into Train and Test Data 
   ````
   X_train,Y_train,X_test,Y_test=train_test_split(X,Y,test_size=0.30)
   ````

## Linear Regression:
In the linear Regression, there is one independent and one dependent variable and the relationship between them is straight line. The equation of regression line is of form y=mx+c, where m is the slope of Regression line, c is intercept, x is independent variable and y is dependent variable. Example - The CGPA being independent variable predicts the package of the student(dependent variable). 
Below is the implementation of Linear Regression Model-
1. Importing the important packages.
   ``````
   import pandas as pd
   import numpy as np
   ``````
2.  Readig the csv file and creating the dataframe of the read data from the csv file using pandas functions.
     ``````
     dataset=pd.read_csv('placement.csv')
     df=pd.DataFrame(dataset)
     df.head(2)
     ``````
3. Setting dependent and independent variables from the dataset.
    `````
    X=df.iloc[:,0:1]
    Y=df.iloc[:,-1]
    `````
4. Importing the train_test_split model and splitting data into train and test set.
    ``````
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)
    ``````
5. Importing the LinearRegression model and creating object of the LinearRegression() type. Afterwards we fit the Training data into the model.
     `````
     from sklearn.linear_model import LinearRegression
     LR=LinearRegression()
     LR.fit(X_train,Y_train)
     `````
6. Giving the test set and predicting its dependent variable values.
   `````
   y_pred=LR.predict(X_test)
   y_pred
   `````
7. Plotting the relationship betweem the Test set (X) value and (Y) value.
   `````
   import matplotlib.pyplot as plt
   plt.scatter(df['cgpa'],df['package'],color='red')
   plt.plot(X_test,y_pred,color='blue')
   `````

After plotting we see that Regression line is linear and relationship between them is straight line.

   ![image](https://github.com/bubblepreetkaur06/learn-python/assets/164672202/e95e26a5-7231-44dc-bb51-9fc9aa5f3245)



     

## Multiple Regression:
In the multiple Regression , there is one dependent variable(y) but many independent variables(x1,x2---xn).
   
