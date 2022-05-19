import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tabulate import tabulate

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn import metrics

date=pd.read_csv('D:\Facultate\Anul 3\TO\\bd_regresie_tema1.csv', usecols = ['bodyfat', 'age', 'cest', 'hip', 'forearm'])

# t = np.array(date['tempo'])
# b = np.array(date['chroma_stft'])
# t=input('Please input feature: ')
# b=input('Please input target: ')
# t=np.fromstring(t,sep=' ',dtype=int)
# b=np.fromstring(b,sep=' ',dtype=int)
# A=np.ones((10,1))
b = date.forearm
b_bar=np.mean(b)

# atribute/predictori/variabile independente:
m = len(date.index)

# t1 = np.array(date['forearm']) # (252,)
# t1 = t1.reshape(m,1) # (252,1)
# t2 = np.array(date['wrist'])
# t2 = t2.reshape(m,1)
# t3 = np.array(date['weight'])
# t3 = t3.reshape(m,1)

# predictor = []
min_SSE = np.inf
min_grad = -1
n = 8

b = np.array(b)
data=[]

def error(b, b_pred):
  SSE = metrics.mean_squared_error(b, b_pred) * m
  MAE = metrics.mean_absolute_error(b, b_pred)
  MSE = metrics.mean_squared_error(b, b_pred)
  RMSE = np.sqrt(metrics.mean_squared_error(b, b_pred))
  r = metrics.r2_score(b, b_pred)
  # b_bar = np.mean(b)
  # SSE = np.linalg.norm(b_pred - b) ** 2
  # r = 1 - SSE / (np.linalg.norm(b - b_bar) ** 2)
  # MAE = np.linalg.norm(b - b_pred, 1) / m
  # MSE = SSE / m
  # RMSE = np.sqrt((MSE))
  data.append([i, r, SSE, MAE, MSE, RMSE])
  return data



for i in range(0, 4):

  t = date.iloc[:, i].to_numpy().reshape(-1, 1)
  coloane=date.columns[i]
  t0 = time.time()
  x_test = np.linspace(np.min(t), np.max(t), 252).reshape(-1, 1)
  for k in range(1, n + 1):  # n gradul pol

    A = np.ones(m).reshape(-1, 1)
    A_test = np.ones(m).reshape(-1, 1)


    for j in range(1, k + 1):  # puterile de la 1 la n

      v = t ** j
      #print(t ** j)

      A = np.hstack((A, t ** j))
      A_test = np.hstack((A_test, x_test ** j))

    x_test = np.linspace(np.min(t), np.max(t), 252).reshape(-1,1)

    polyreg = make_pipeline(PolynomialFeatures(k), LinearRegression())
    polyreg2 = make_pipeline(PolynomialFeatures(k), LinearRegression())
    A= A[:,1:]
    A_test= A_test[:,1:]

    polyreg.fit(A,b)
    polyreg2.fit(A_test,b)

    lin = LinearRegression()
    lin2 = LinearRegression()
    lin.fit(A,b)
    lin2.fit(A_test,b)

    #print("coefs",np.shape(coefs),"a",np.shape(A))
    b_pred = lin.predict(A)
    b_pred2 = lin.predict(A_test)

    #
    # # plt.subplot(4,2,k)
    # # plt.imshow(line,predict(line))
    # plt.subplot(1, 2, 1)
    # plt.scatter(t, b, 2)
    # plt.title("Linear Regression")
    # # print(len(b)," B ",len(x_test), " x ")
    # plt.plot(x_test, b_pred2, 'r-')
    # plt.show()



    # poly = PolynomialFeatures(degree = 8)
    # t_poly = poly.fit_transform(t)
    # poly.fit(t_poly,b)
    # lin2 = LinearRegression()
    # lin2.fit(t_poly,b)
    # b_pred = lin2.predict(poly.fit_transform(t))
    # # lin = LinearRegression()
    # #
    # # lin.fit(t[:,np.newaxis],b)
    # x_test = np.linspace(np.min(t), np.max(t), 252)
    # #x_plot = poly.fit_transform(x_test)
    # # y_pred = lin.predict(x_test)

    # SSE = metrics.mean_squared_error(b, b_pred)*m
    # MAE = metrics.mean_absolute_error(b, b_pred)
    # MSE = metrics.mean_squared_error(b,b_pred)
    # RMSE = np.sqrt(metrics.mean_squared_error(b, b_pred))
    # r = metrics.r2_score(b, b_pred)

    data_linear = error(b,b_pred)
    #data.append([k,r,SSE,MAE,MSE,RMSE])

    # model = np.polyfit(t.flatten(),b,k)
    # predict = np.poly1d(model)
    # line = np.linspace(min(t),max(t),255)
    # t_idx = np.argsort(t)
    # #coefs = np.hstack((lin.coef_, lin.intercept_))
    # t = np.sort(t)
    #
    # y_test1 = np.polyval(x_test, x_test)
    # b = b[t_idx]

    #
    # # plt.subplot(4,2,k)
    # # plt.imshow(line,predict(line))
    #plt.subplot(1, 2, 2)
    plt.scatter(t, b, 2)
    plt.title("Polynomial Features" )
    # print(len(b)," B ",len(x_test), " x ")
    plt.plot(x_test,b_pred2,'r-')
    plt.suptitle("Gradul = {grad} Predictor = {col}".format(grad=k,col=coloane))
    plt.show()
  print("----------------------",coloane,"--------------")
  print(tabulate(data_linear, headers=["Grad", "R^2", "SSE", "MAE", "MSE", "RMSE"]))
  data.clear()
  t1 = time.time()

  print("Timpul executie pentru ",coloane," este ", t1-t0)

