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



data=[]
  # 00 10 01 + 20 + 11 + 02
m = len(date.index)
b=date.forearm
# atribute/predictori/variabile independente:
t1 = date.iloc[:,0].to_numpy().reshape(-1,1)
t2 = date.iloc[:,3].to_numpy().reshape(-1,1)
b=np.array(b).reshape(-1,1)
print("m " , m,"t1 " ,len(t1)," t2 ",len(t2)," b " ,len(b))
t0 = time.time()
# A=np.ones((m,1))
grad = 8
data=[]

def matrice(t1,t2,grad):
  j=0
  sum=0
  A = np.ones((m,1))
  for k in range(0,grad+1):
    for i in range(0,k+1):
      sum=sum+((t1 ** (k-i)) * (t2**i))
      A=np.hstack((A,sum))
      # print("j",j)
      print("i",i)
      string = " * t1 **" + str((k - i)) + " * t2 ** " + str(i)
      string += string

      j = j + 1
      # print(A)
    print("k",k)
  #print(A)
  print(string)
  return A
def functie(t1,t2,grad,x):
  j=0
  sum=0
  for k in range(0,grad+1):
    for i in range(0,k+1):
      sum=sum + x[j] * (t1 ** (k-i)) * (t2**i)
      string="x"+ str(j)+ " * t1 **"+ str((k-i))+ " * t2 ** "+str(i)
      string +=string
      print("x",j," * t1 **",(k-i)," * t2 ** ",i)
      # print("j",j)
      print("i",i)
      j = j + 1
    print("k",k)
  print(string)
  return sum
def error(b, b_pred,i):
  # SSE = metrics.mean_squared_error(b, b_pred) * m
  # MAE = metrics.mean_absolute_error(b, b_pred)
  # MSE = metrics.mean_squared_error(b, b_pred)
  # RMSE = np.sqrt(metrics.mean_squared_error(b, b_pred))
  # r = metrics.r2_score(b, b_pred)
  b_bar = np.mean(b)
  SSE = np.linalg.norm(b_pred - b) ** 2
  r = 1 - SSE / (np.linalg.norm(b - b_bar) ** 2)
  MAE = np.linalg.norm(b - b_pred, 1) / m
  MSE = SSE / m
  RMSE = np.sqrt((MSE))
  data.append([i, r, SSE, MAE, MSE, RMSE])
  return data
def plotare(coefs,grad):
  fig = plt.figure()
  ax = plt.axes(projection='3d')

  ax.scatter3D(t1, t2, b)

  t1_plt = np.linspace(np.min(t1), max(t1), m)
  t2_plt = np.linspace(np.min(t2), max(t2), m)

  X, Y = np.meshgrid(t1_plt, t2_plt)
  Z = functie(X, Y, grad,coefs)
  #print("X", len(X), X.ndim, "\n Y: ", len(Y), Y.ndim, "\n coefs : ", len(coefs), coefs.ndim)

  ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

  plt.suptitle("Gradul = {grad}".format(grad=grad))
  plt.show()
  ax.view_init(60, 35)
  plt.show()



def model(i):
  A=matrice(t1,t2,i)

  A = A[:, 1:] # pentru LinearRegression()
  X_train, X_test, y_train, y_test = train_test_split(A, b, shuffle=True, train_size=0.5)
  modelul_nostru = LinearRegression().fit(X_train,y_train)
  coefs = np.hstack((modelul_nostru.coef_[0,::-1],modelul_nostru.intercept_))
# y_pred = modelul_nostru.predict(X_train) # pentru datele de training; trebuie sa coincida cu X_train * coefs
  y_pred2 = modelul_nostru.predict(X_test) # pentru datele de test; trebuie sa coincida cu X_test * coefs

  x = np.linalg.solve(np.dot(A.T,A),np.dot(A.T,b))
  data=error(y_test,y_pred2,i)
  print(tabulate(data, headers=["Grad", "R^2", "SSE", "MAE", "MSE", "RMSE"]))

  plotare(x,i)









for i in range(1,9):
  model(i)
t1 = time.time()

print("Timpul executie pentru gradul ", grad, " este ", t1 - t0)
