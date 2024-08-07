#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#First load csv file


# In[1]:


import pandas as pd


# In[12]:


import os


# In[10]:


os.getcwd ()


# In[26]:


df = pd.read_excel(r"F:\WORK\Ac\DATA ANALYTICS - IIT ROORKEE\Copy of Data Analytics minor project-Vehical insurance prediction F.xlsx")


# In[29]:


df.head ()


# In[30]:


df.tail ()


# In[31]:


df.tail(12)


# In[32]:


df.shape


# In[ ]:


#the field here shows total no. of rows and columns in the data set


# In[34]:


df.columns


# In[35]:


df['Vehicle_Damage'].replace(['Yes','No'],[1,0],inplace =True)


# In[ ]:


#here the entries under Vehicle_Damage are replaced by numeric value


# In[36]:


df


# In[37]:


df.isnull().sum()


# In[ ]:


#output shows there is no null value in any entry of any columns


# In[38]:


df.info()


# In[ ]:


#shows the data type 


# In[39]:


df.describe()


# In[ ]:


#shows descriptive statistical data for numerical data columns


# In[40]:


df[df.duplicated()]


# In[44]:


df['Response'].value_counts()


# In[2]:


import seaborn as sns


# In[4]:


sns.set(color_codes = True)


# In[5]:


df = pd.read_excel(r"F:\WORK\Ac\DATA ANALYTICS - IIT ROORKEE\Copy of Data Analytics minor project-Vehical insurance prediction F.xlsx")


# In[17]:


sns.distplot(df['Age'])


# In[ ]:


#distplot used for single column distribution


# In[14]:


df


# In[23]:


sns.jointplot(df['Annual_Premium'],df['Age'])


# In[15]:


df['Vehicle_Age'].value_counts()


# In[24]:


df['Vehicle_Damage'].replace(['Yes','No'],[1,0],inplace =True)


# In[25]:


df


# In[27]:


df['Vehicle_Damage'].value_counts()


# In[37]:


sns.jointplot(df['Annual_Premium'],kind = "hex")


# In[ ]:


sns.jointplot(df['Annual_Premium'],kind = "kde")


# In[ ]:


#kde here gives kernel graphs 9There are 2 kinds "kde" for kernel and "hex" for hexagon symbol insted of point or dot


# In[ ]:


sns.pairplot(df[['Age','Annual_Premium', 'Vehicle_Damage']])


# In[49]:


sns.stripplot (df['Annual_Premium'], jitter = True)


# In[ ]:


#strip plot likewise 3 different arguments which will give 3 strips 


# In[ ]:


# now let's analyze data using basic functions


# In[54]:


df.dtypes


# In[57]:


df['Gender'].unique()


# In[58]:


df


# In[59]:


df['Region_Code'].unique()


# In[66]:


df['Region_Code'].nunique()


# In[60]:


df.nunique()


# In[ ]:


# gives the ist of all unique value in each column


# In[61]:


df.count()


# In[62]:


df['Gender'].value_counts()


# In[63]:


df['Age'].value_counts()


# In[64]:


df['Vehicle_Damage'].value_counts()


# In[ ]:


# like this we can analyze each colum head, data entry counts specifically


# In[ ]:


# now we are filtering data for only values with Male under particular column head


# In[71]:


df[df.Gender == 'Male']


# In[ ]:


# OR CAN USE groupby command


# In[77]:


df.groupby('Gender').get_group('Male')


# In[ ]:


# now we are filtering data for only values with 1 under Response column head


# In[80]:


df[df['Response'] == 1]


# In[82]:


df.notnull().sum()


# In[83]:


df.rename(columns = {'Gender' : 'Sex'})


# In[ ]:


#here we have successfully changed the column head name


# In[ ]:


# But this is for temeporary purpose as we can see below


# In[84]:


df


# In[94]:


df.rename(columns = {'Gender':'Sex'})


# In[ ]:


#To make permanent change add inplace = True


# In[ ]:


# now to ind mean value of any column value for analysis


# In[95]:


df.Annual_Premium.mean()


# In[ ]:


# to find mean age of respondents


# In[96]:


df.Age.mean()


# In[ ]:


# likewise standard deviation of Annual_Premium


# In[97]:


df.Annual_Premium.std()


# In[ ]:


# likewise variance of Annual_Premium


# In[98]:


df.Annual_Premium.var()


# In[ ]:


# however in the above command we use [] if the column head name has space in btw


# In[ ]:


# now using 'and' operator if we want to filer two things


# In[99]:


df.head (2)


# In[ ]:


# Now we want data in which age is more than 44 and response is 1 by using & command


# In[106]:


(df['Age']>44) & (df['Response'] == 1)


# In[121]:


columns_to_drop = ['Vehicle_Age', 'Unnamed: 12', 'Unnamed: 13' ]
df_dropped = df.drop(columns=columns_to_drop, axis=1)


# In[122]:


df_dropped


# In[ ]:


# mean value of each column against Gender


# In[125]:


df_dropped.groupby('Gender').mean()


# In[ ]:


# min and max value of each column against gender


# In[126]:


df_dropped.groupby('Gender').min()


# In[127]:


df_dropped.groupby('Gender').max()


# In[ ]:


# now finding no. of Male with Response 1 i.e. Yes


# In[128]:


df_dropped[(df['Response'] == 1) & (df['Gender'] == 'Male')]


# In[ ]:


# now finding no. of Age is 47  with Response 1 i.e. Yes


# In[131]:


df_dropped[(df['Response'] == 1) & (df['Age'] == '47')]


# In[ ]:


# THANKYOU

