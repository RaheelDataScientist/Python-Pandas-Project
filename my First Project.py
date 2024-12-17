#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
df = pd.read_csv(r'C:\Users\DELL\Downloads\Indian Liver Patient Dataset (ILPD).csv')
df


# In[21]:


df.shape


# In[22]:


df.shape[0]


# In[23]:


df.shape[1]


# In[24]:


df.head()


# In[25]:


df.tail()


# In[26]:


df.describe()


# In[27]:


df.info()


# In[28]:


df.isnull()


# In[33]:


df.isnull().sum()


# In[32]:


df['alkphos'].ffill(inplace = True)


# In[31]:


df


# # Data Visualization

# In[34]:


import matplotlib.pyplot as plt


# In[37]:


df.plot()


# In[38]:


df.plot(subplots = True)                                                     


# In[68]:


df.plot(kind = 'bar')                        


# In[69]:


df.plot(kind = 'bar', subplots = True)


# In[70]:


df.plot(kind = 'bar', subplots = True, figsize = (12, 3))


# In[72]:


import pandas as pd


data = {'Category': ['A', 'B', 'C', 'D'],
        'Values': [10, 20, 30, 40]}
df = pd.DataFrame(data)


df.set_index('Category')['Values'].plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))

plt.ylabel('')  
plt.show()


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100),
    'D': np.random.randn(100)
}
df = pd.DataFrame(data)
df.plot(kind='density', subplots=True, layout=(2, 2), figsize=(10, 6))
plt.show()


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Q1': [1, 2, 3, 4, 5],
    'T1': [10, 20, 25, 30, 50]
})


plt.scatter(data['Q1'], data['T1'])
plt.xlabel('Heater (%)')
plt.ylabel('Temperature (°C)')
plt.show()


# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [5, 4, 3, 2, 1]
})

corr = data.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Heatmap')
plt.show()


# In[7]:


import seaborn as sns
sns.pairplot(data)


# In[8]:


import seaborn as sns
sns.heatmap(data.corr())


# In[ ]:




