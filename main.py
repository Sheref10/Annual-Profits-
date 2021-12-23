import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as pyp
import operator

data=pd.read_csv('Task2_Poly_1.csv')
data_real=pd.read_csv('Task2_Poly_2.csv')
x=data.iloc[:,0:1]
y=data.iloc[:,1]
y_real=data_real.iloc[:,1]

poly=PolynomialFeatures(degree=5)
x_poly=poly.fit_transform(x)
model=LinearRegression()
num=np.array(range(5,101,5)).reshape(-1,1)
model.fit(x_poly,y)
miss_poly=poly.fit_transform(num)
y_predicted=model.predict(miss_poly)
print(y_predicted)
MSE=mean_squared_error(y_real,y_predicted)
r2=r2_score(y_real,y_predicted)
print("RMSE: ",MSE)
print("R2: ",r2)
pyp.scatter(x, y)

