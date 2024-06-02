# Regression Models
Regression is a technique to examine the relationship between the Dependant variable (Y) and Independent Variables(X).
It is a type of supervised Learning.
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

There are many types of Regression models:
1. Linear Regression Model
2. Multiple Regression Model
3. Logistic Regression Model
4. Ridge Regression Model.
5. Lasso Regression Model
   
