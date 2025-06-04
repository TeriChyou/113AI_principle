import numpy as np
import pandas as pd
df=pd.read_csv('iris_total/iris.data', header=None)
#print(df)
#x=df.iloc[0:100,[0,1,2,3]].values
x=df.iloc[0:100,0:4].values
y=df.iloc[0:100,4].values
y_onehot=pd.get_dummies(y)
print(y_onehot)
#x=df.iloc[0:150,[0,1,2,3]].values
#y=df.iloc[0:150,4].values
#numpy.where(condition, [x, y, ]/) 
#Return elements chosen from True x or False y depending on condition.
y=np.where(y=='Iris-setosa', 0, 1)
#y=np.where(y=='Iris-setosa', 0, y)
#y=np.where(y=='Iris-versicolor', 1, y)
#y=np.where(y=='Iris-virginica', 2, y)
print(y)

x_train=np.empty((80,4))
x_test=np.empty((20,4))
y_train=np.empty(80)
y_test=np.empty(20)
x_train[:40],x_train[40:]=x[:40],x[50:90]
x_test[:10],x_test[10:]=x[40:50],x[90:100]
y_train[:40],y_train[40:]=y[:40],y[50:90]
y_test[:10],y_test[10:]=y[40:50],y[90:100]
#print(y_train)

#define function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(x,w,b):
    return sigmoid(np.dot(x,w)+b)

def update(x,y,w,b,eta):
    y_pred=activation(x, w, b)
    a=(y_pred-y_train)*y_pred*(1-y_pred)
    for i in range(4):
        w[i]-=eta*1/float(len(y))*np.sum(a*x[:,i])
    b-=eta*1/float(len(y))*np.sum(a)
    return w,b


#the initial value of w,b and eta
weights=np.ones(4)/10
bias=np.ones(1)/10
eta=0.1
#print(bias)
for _ in range(100):
    weights,bias=update(x_train,y_train,weights,bias,eta)
    
#print('weights=',weights,'bias=',bias)
#use test data to see the result
print(activation(x_test, weights, bias))
