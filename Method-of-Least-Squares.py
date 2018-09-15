import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from numpy.linalg import inv

def my_fit(A, y):
    A['const'] = 1
    estimator = np.matmul(np.matmul(inv(np.matmul(A.T, A)), A.T), y)
    A = A.drop(['const'], axis=1)
    return estimator

df = pd.read_csv('HousePriceData.csv')  # Data taken from a Kaggle competition
                                        # I'm competing in https://goo.gl/qBhxUd

House_Feature_Data = df.drop(['SalePrice'],  axis=1)
House_Sale_Price_Data = df['SalePrice']


# Testing against machine-learning library scikit-learn# Testing 

ols = linear_model.LinearRegression().fit(House_Feature_Data, House_Sale_Price_Data)
my_ols = my_fit(House_Feature_Data, House_Sale_Price_Data)

print('sklearn\'s intercept is: ', ols.intercept_)
print('my algo\'s intercept is: ', my_ols[my_ols.size - 1])
print('sklearn\'s beta estimator is: ', ols.coef_)
print('my algo\'s beta estimator is: ', my_ols[0:my_ols.size - 1])