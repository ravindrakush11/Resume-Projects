#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np


# In[7]:


bc = datasets.load_breast_cancer()
bc


# In[3]:


X = bc.data  # slicing for selecting features [: [2,3]]
Y = bc.target
print("Input Instances: ")
print(X)
print()
print("Target Values: ")
print(Y)
print()
print("Unique Values: ")
print(np.unique(Y))


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 0)
print("Complete Data: ")
print(np.shape(X))
print()
print("Training Data: ")

print(np.shape(X_train))
print()
print("Testing Data: ")

print(np.shape(X_test))


# In[37]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_sc = sc.transform(X_train) 
X_test_sc = sc.transform(X_test)

print(X_train_sc)
print()
print(X_test_sc)


# In[38]:


from sklearn.linear_model import Perceptron

per = Perceptron(eta0 = 0.35, random_state = 0)
per.fit(X_train_sc, Y_train)
Y_pred = per.predict(X_test_sc)

print("Total number of misclassifications: ")
print((Y_test != Y_pred).sum())


# In[40]:


from sklearn.metrics import accuracy_score
print("Accuracy")
accuracy_score(Y_test, Y_pred)

