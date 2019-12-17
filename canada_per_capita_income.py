#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('canada_per_capita_income.csv')


# In[3]:


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values


# In[4]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)


# In[5]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)


# In[6]:


plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[7]:


plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:




