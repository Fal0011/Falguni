#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load libraries


# In[1]:


import pandas as pd 


# In[2]:


import matplotlib.pyplot as plt 


# In[3]:


import numpy as np


# In[4]:


import seaborn as sns


# In[5]:


import plotly.express as px


# In[6]:


sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})


# In[7]:


from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support as score, roc_curve


# In[8]:


from sklearn.model_selection import cross_val_score, train_test_split, cross_validate


# In[9]:


from sklearn.utils import compute_sample_weight


# In[10]:


from xgboost import XGBClassifier


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


pip install xgboost


# In[13]:


df = pd.read_csv(r"F:\WORK\Ac\DATA ANALYTICS - IIT ROORKEE\Data Analytics Machine failure prediction - Major project\Machine failure prediction - Major project\data (1).csv") 


# In[14]:


df


# In[15]:


df_original = pd.DataFrame(df)


# In[16]:


df


# In[17]:


sns.countplot(x='fail', data=df)
plt.show()


# In[18]:


print(df.shape[0])
df.head(3)


# In[19]:


df.isnull().sum()


# In[20]:


df.info()


# In[21]:


# check missing values 


# In[22]:


def print_missing_values(df):
    null_df = pd.DataFrame(df.isna().sum(), columns=['null_values']).sort_values(['null_values'], ascending=False)
    fig = plt.subplots(figsize=(16, 6))
    ax = sns.barplot(data=null_df, x='null_values', y=null_df.index, color='royalblue')
    pct_values = [' {:g}'.format(elm) + ' ({:.1%})'.format(elm/len(df)) for elm in list(null_df['null_values'])]
    ax.set_title('Overview of missing values')
    ax.bar_label(container=ax.containers[0], labels=pct_values, size=12)


# In[23]:


num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


# In[24]:


cor_matrix = df[num_features].corr()
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[25]:


df


# In[26]:


if df.isna().sum().sum() > 0:
    print_missing_values(df)
else:
    print('no missing values')


# In[27]:


# drop all columns with more than 5% missing values


# In[28]:


for col_name in df.columns:
    if df[col_name].isna().sum()/df.shape[0] > 0.05:
        df.drop(columns=[col_name], inplace=True) 


# In[29]:


df.columns


# In[30]:


df.head()


# In[31]:


target_name='fail'


# In[32]:


# display class distribution of the target variable


# In[33]:


px.histogram(df, y="fail", color="fail") 


# In[34]:


cor_matrix = df[num_features].corr()
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[35]:


sns.pairplot(df, height=2.5, hue='fail')


# In[36]:


# correlation plot


# In[37]:


plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), cbar=True, fmt='.1f', vmax=0.8, annot=True, cmap='Blues')


# In[38]:


# create histograms for feature columns separated by target column


# In[39]:


df


# In[40]:


def create_histogram(column_name):
    plt.figure(figsize=(16,6))
    return px.box(data_frame=df, y=column_name, color='fail', points="all", width=1200)


# In[41]:


create_histogram('tempMode')


# In[42]:


create_histogram('AQ')


# In[43]:


create_histogram('USS')


# In[44]:


create_histogram('CS')


# In[45]:


create_histogram('VOC')


# In[46]:


create_histogram('RP')


# In[47]:


create_histogram('IP')


# In[48]:


create_histogram('Temperature')


# In[49]:


target_name='fail'


# In[50]:


def data_preparation(df, target_name):
    df = df.dropna()


# In[51]:


X = df.drop(columns=[target_name])
y = df[target_name]


# In[52]:


df


# In[53]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

label_encoder.fit(df['footfall'])
df['footfall'] = label_encoder.transform(df['footfall'])

label_encoder.fit(df['fail'])
df['machine failure'] = label_encoder.transform(df['fail'])


# In[54]:


# split the data into x_train and y_train data sets


# In[55]:


X = df.drop(['fail'],axis=1)
y = df['fail']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42, test_size = 0.20)


# In[56]:


print('train: ', X_train.shape, y_train.shape)
print('test: ', X_test.shape, y_test.shape)


# In[57]:


from sklearn.linear_model import LogisticRegression


# In[59]:


pip install scikit-learn


# In[62]:


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[74]:


import plotly.express as px

fig = px.scatter_3d(df, x='VOC', y='RP', z='IP',
              color='fail')
fig.show()


# In[60]:


# Checking the accuracy


# In[64]:


from sklearn.metrics import accuracy_score


# In[65]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[66]:


import matplotlib.pyplot as plt
#Graphical representation of confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




