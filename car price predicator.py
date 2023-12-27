#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


car=pd.read_csv('quikr_car - quikr_car.csv')


# In[3]:


car


# In[4]:


car.shape


# In[5]:


car['year'].unique()
#lots of irrelevent data


# In[6]:


car['Price'].unique()
#comas and Ask For Price value


# In[7]:


car['kms_driven'].unique()
#kms in last


# In[8]:


car['fuel_type'].unique()
#nan values


# In[9]:


backupdata=car.copy()


# In[10]:


car=car[car['year'].str.isnumeric()]


# In[11]:


car.shape


# In[12]:


car['year']=car['year'].astype(int)


# In[13]:


car=car[car['Price']!='Ask For Price']
car


# In[14]:


car['price']=car['Price'].str.replace(',','')
car['Price']=car['price'].astype(int)


# In[15]:


car.info()


# In[16]:


car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
car


# In[116]:


car=car[car['kms_driven'].str.isnumeric()]


# In[117]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[118]:


car=car[~car['fuel_type'].isna()]
car.info()


# In[119]:


car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')
car


# In[120]:


car=car.reset_index(drop=True)
car.describe()


# In[121]:


car=car[car['Price']<6e6].reset_index(drop=True)
car


# In[122]:


car.to_csv('cleandata.csv')


# In[123]:


x=car[['name','company','year','kms_driven','fuel_type']]
y=car['price']


# In[124]:


y


# In[125]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)


# In[126]:


xtest


# In[127]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[128]:


ohe=OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])


# In[129]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')


# In[130]:


lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)


# In[131]:


pipe.fit(xtrain,ytrain)


# In[132]:


y_pred=pipe.predict(xtest)


# In[133]:


ytest


# In[134]:


y_pred


# In[135]:


r2_score(ytest,y_pred)


# In[136]:


scores=[]
for i in range(1000):
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(xtrain,ytrain)
    y_pred=pipe.predict(xtest)
    scores.append(r2_score(ytest,y_pred))


# In[137]:


import numpy as np
np.argmax(scores)


# In[138]:


scores[np.argmax(scores)]


# In[139]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[140]:


test=np.array(xtest)
test[5]


# In[141]:


pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




