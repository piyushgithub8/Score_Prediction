<h1><center>Score Prediction By The Amount of Hours Studied</center></h1>

![Studying-56a945f83df78cf772a55e31.jpg](attachment:Studying-56a945f83df78cf772a55e31.jpg)

<h2>Introduction</h2>
In the following notebook, we will see the scores of student based on the hours of study. This is a simple linear regression notebook.


<h2>Features</h2>
<ul>
    <li><b>Hours  : </b> Number of hours studied
    <li><b>Scores : </b> Resulting score after studying 

<h2>Objective</h2>
To predict the score by using the number of hours studied. 

<h3>Importing Libraries</h3>


```python
#importing requiring libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
```

<h3>Reading & Loading Dataset</h3>


```python
#reading the dataset
study=pd.read_excel('Study_hours.xlsx')
```


```python
#getting the shape of the dataset
study.shape
```




    (25, 2)




```python
#getting peek of the first five observations
study.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
#getting general information about the features of the dataset
study.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 25 entries, 0 to 24
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Hours   25 non-null     float64
     1   Scores  25 non-null     int64  
    dtypes: float64(1), int64(1)
    memory usage: 528.0 bytes
    


```python
#getting peek count, mean, min, max, std of the features
study.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>25.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.012000</td>
      <td>51.480000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.525094</td>
      <td>25.286887</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.100000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.700000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.800000</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.400000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.200000</td>
      <td>95.000000</td>
    </tr>
  </tbody>
</table>
</div>



<li> The reading and loading of dataset has been performed. There are two features in the dataset. The data type of the feature is float and integer. There are total 25 number of observations(rows).

<h3>Presence of Null Values</h3>


```python
#checking presence of null values in the dataset
study.isnull().sum()
```




    Hours     0
    Scores    0
    dtype: int64



<li> The dataset does not contain any null values.

<h3>Dropping Duplicates</h3>


```python
#dropping duplicate values present in the dataset
study.drop_duplicates(inplace=True)
```

<li>Duplicate observations has been dropped. 

<h3>Feature Distribution</h3>


```python
#checking the distribution of the Hours feature
plt.figure(figsize=(7,6))
sns.distplot(study['Hours'])
plt.title('Hours Distrubution')
plt.grid()
plt.show()
```


    
![png](output_21_0.png)
    



```python
#checking the distribution of the Scores feature
plt.figure(figsize=(7,6))
sns.distplot(study['Scores'])
plt.title('Scores Distribution')
plt.grid()
plt.show()
```


    
![png](output_22_0.png)
    


<li> The Independent and Dependent features of the dataset are fairly distributed.

<h3>Relation Between Features</h3>


```python
#checking the relation between hours and scores 
sns.scatterplot(data=study,x='Hours',y='Scores')
plt.title('Relation Between Hours and Scores')
plt.grid()
plt.show()
```


    
![png](output_25_0.png)
    


<li>After plotting Hours and Scores in a scatter plot, a linear relation has been observed between Hours and Scores. 
<li>This scatter plot concludes that, as the hour of study increases the score also increases.  

<h3>Splitting Dependent and Independent Features</h3>


```python
#splitting the dataset
X=study.iloc[:,:1]
Y=study.iloc[:,-1:]
```

<h3>Train-Test Split</h3>


```python
#train and test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)
```

<h3>Model Training</h3>


```python
#importing libraries for model training
from sklearn import metrics
from sklearn.linear_model import LinearRegression
```


```python
#linear Regression model
lr=LinearRegression()
```


```python
#fitting the train and test dataset into linear Regression model
lr.fit(X_train,Y_train)
```




    LinearRegression()




```python
#gives the coefficient of the features of your dataset
m=lr.coef_
print('The coefficient value : ',m)
#represents the mean value of the response variable when all of the predictor variables in the model are equal to zero
c=lr.intercept_
print('The intercept value : ',c)
```

    The coefficient value :  [[9.60498629]]
    The intercept value :  [3.09804089]
    

<h3>Prediction and Error</h3>


```python
#predicting y using the test data
y_pred=lr.predict(X_test)
#calculating error
error=Y_test - y_pred
```


```python
#checking the score of train data
train_score=lr.score(X_train,Y_train)
print('The train score : ',train_score)
#checking the score of test data
test_score=lr.score(X_test,Y_test)
print('The test score : ',test_score)
```

    The train score :  0.9583509805345388
    The test score :  0.9000546706590146
    

<li>The train set has performed with an accuracy of 95%
<li>The test set has performed with an accuracy of 90%

<h3>Model Evaluation</h3>


```python
#import model evaluation parameters
from sklearn.metrics import mean_absolute_error,mean_squared_error
```


```python
#model evaluation through mean absolute error method
print('MAE : ',mean_absolute_error(Y_test,y_pred))
#model evaluation through mean squared error method
print('MSE : ',mean_squared_error(Y_test,y_pred))
#model evaluation through root mean squared error
print('RMSE : ',np.sqrt(mean_squared_error(Y_test,y_pred)))
```

    MAE :  6.012413762390567
    MSE :  37.56345257951597
    RMSE :  6.1289030486308045
    

The Model Evaluation has been performed and the Mean Absolute Error is 6 which is inclined towards zero.


```python
#pltting regression line across the points of 
regression_line= m * X + c
plt.figure(figsize=(8,6))
plt.scatter(X,Y)
plt.plot(X,regression_line)
plt.title('Best Fit Line')
plt.grid()
plt.show()
```


    
![png](output_44_0.png)
    


This scatter plot represents the regression line and the actual data points. From the graph it can be seen that the error(residual) is very less and the model is  ready. 

<h3>Score Prediction</h3>

<li>Using the coefficient and the intercept to predict the score:


```python
#creating a function to predict scores based on number of hours 
def predit_score(h):
    score=(m*h+c)
    print(f'The Score after studying for {h} hours would be ',round(int(score)))
```


```python
#score predictor function
predit_score(9.25)
```

    The Score after studying for 9.25 hours would be  91
    

The score has been sucessfully predicted. 
