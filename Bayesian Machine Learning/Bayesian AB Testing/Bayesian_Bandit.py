#!/usr/bin/env python
# coding: utf-8

# In[35]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
import seaborn as sns
sns.set()


# In[36]:


NUM_T = 2000
BANDIT_P = [0.2,0.5,0.75]


# In[37]:


class Bandit:
    def __init__(self,p):
        self.p = p
        self.a = 1
        self.b = 1
    
    def pull(self):
        return np.random.random() < self.p
    
    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1-x
        


# In[38]:


def plot(bandits, trail):
    x = np.linspace(0,1,200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x,y,label='real p:  %.4f' % b.p)
    plt.title('Bandit distribution after %s trails' % trail)
    plt.legend()
    plt.show()


# In[39]:


def experiment():
    bandits = [Bandit(p) for p in BANDIT_P]
    
    sample_points = [5,10,20,50,100,500,1000, 1999]
    for i in range(NUM_T):
        bestb = None
        maxsample = -1
        allsamples = []
        for b in bandits:
            sample = b.sample()
            allsamples.append('%.4f' % sample)
            if sample > maxsample:
                maxsample = sample
                bestb = b
        if i in sample_points:
            print('current samples:' % allsamples)
            plot(bandits, i)
                
        x = bestb.pull()
        bestb.update(x)


# In[40]:

if __name__ == "__main__":
  experiment()


# In[ ]:




