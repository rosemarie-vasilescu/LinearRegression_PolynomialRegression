import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
from tabulate import tabulate


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
for i in range(0, 4):

  t = date.iloc[:, i].to_numpy().reshape(-1, 1)
  coloane=date.columns[i]
  t0 = time.time()
  for k in range(1, n + 1):  # n gradul pol

    A = np.ones(m).reshape(-1, 1)

    for j in range(1, k + 1):  # puterile de la 1 la n

      v = t ** j
      #print(t ** j)

      A = np.hstack((A, t ** j))
      # A=np.hstack((A,t2))
      # A=np.hstack((A,t3))
      # A = np.vander(t, N=i, increasing=True)
      # A=np.insert(A,i,t**i,axis=1)
    # A=np.insert(A,2,t**2,axis=1)
    x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))  # ec normala A.t*A*x=A.t*b


    #y_pred = np.polyval(x[::-1], t)
    y_pred = np.dot(A,x)
    SSE = np.linalg.norm(y_pred - b) ** 2  # suma patratelor erorilor
    if min_SSE > SSE:
      min_SEE = SSE
      min_grad = n
    r = 1-SSE/(np.linalg.norm(b-b_bar)**2)
    print(r)
    MAE = np.linalg.norm(b - y_pred, 1) / m
    MSE = SSE / m
    RMSE = np.sqrt((MSE))


    data.append([k,r,SSE,MAE,MSE,RMSE])

    model = np.polyfit(t.flatten(),b,k)
    predict = np.poly1d(model)
    line = np.linspace(min(t),max(t),255)
    plt.scatter(t,b,2)
    # plt.subplot(4,2,k)
    # plt.imshow(line,predict(line))
    plt.title("Grad {grad} Coloana {col}".format(grad=k,col=coloane))
    #plt.plot(line,predict(line),'r-')
    plt.show()
  print("----------------------",coloane,"--------------")
  print(tabulate(data, headers=["Grad", "R^2", "SSE", "MAE", "MSE", "RMSE"]))
  data.clear()
  t1 = time.time()

  print("Timpul executie pentru ",coloane," este ", t1-t0)