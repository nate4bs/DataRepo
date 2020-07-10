#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[12]:


df.describe()


# In[22]:


Scaled_Amount = df[['Amount']]


# In[23]:


scaler = StandardScaler().fit(Scaled_Amount.values)


# In[24]:


Scaled_Amount = scaler.transform(Scaled_Amount.values)


# In[27]:


df['Amount'] = Scaled_Amount


# In[29]:


X = df.drop(columns = 'Class')
Y = df['Class']


# In[36]:


seed = 7
test_size = 0.2
X_train ,X_test, Y_train, Y_test = train_test_split(X,Y, test_size =test_size, random_state = seed)


# In[37]:


model = XGBClassifier()
model.fit(X_train,Y_train)


# In[ ]:


Y_pred = model.predict(X_test)
predictions = [round(value) for value in Y_pred]


# In[ ]:


accuracy = accuracy_score(Y_test, predictions)
print('Accuracy: %.2f%%' % (accuracy*100.0))

