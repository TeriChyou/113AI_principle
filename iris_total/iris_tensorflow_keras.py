#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


iris = load_iris()
x, y = iris.data, iris.target
print('x=', x)
print('y=', y)
print('x.shape original=', x.shape)


# In[3]:


x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.shape reshape after=', x.shape)
#print(unique(y))
#print(unique(y).sum())


# In[4]:


#split the data into the train and  20% for test part
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2)


# In[5]:


#define the model
model = Sequential()


# In[6]:


#output filters=64,   kernel_size=2
#input_shape argument (tuple of integers or None, 
#e.g. (10, 128) for sequences of 10 vectors of 128-dimensional vectors
#in this example: (x0,x1,x2,x3) 4 column, 1 dimensional vector 
#x.shape reshape after= (150, 4, 1) then input_shape=(4,1)
layer_1 = Dense(16, input_shape=(4,1), activation="relu") # use Dense as input
#layer_1 = Conv1D(64, 2, activation="relu", input_shape=(4,1)) #use filter as layer_1 input 
model.add(layer_1)
#16 neurons
model.add(Dense(16, activation="relu"))
#model.add(MaxPooling1D())
model.add(Flatten())
# 3 type of output
model.add(Dense(3, activation = 'softmax'))


# In[7]:


#compile the model
model.compile(loss = 'sparse_categorical_crossentropy',  optimizer = "adam",  metrics = ['accuracy'])
#model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[8]:


model.summary()


# In[9]:


#fit the model with train data
model.fit(xtrain, ytrain, batch_size=16,epochs=100, verbose=0)


# In[10]:


#calculate the loss and accuracy with train data
loss, accuracy = model.evaluate(xtrain, ytrain)
print("Loss:", loss, " Accuracy:", accuracy)


# In[11]:


#calculate the loss and accuracy with test data
loss, accuracy = model.evaluate(xtest, ytest)
print("Loss:", loss, " Accuracy:", accuracy)


# In[12]:


#predict the test data
pred = model.predict(xtest)
#print(pred)
pred_y = pred.argmax(axis=-1)


# In[13]:


#confusion matrix
cm = confusion_matrix(ytest, pred_y)
print('cm=',cm)


# In[14]:


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()

