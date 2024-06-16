#!/usr/bin/env python
# coding: utf-8

# # Data Salary Prediction Model

# ## Package Imports

# In[1]:


import pandas as pd
import tensorflow_decision_forests as tfdf


# ## Prepare Dataset

# In[2]:


dataset = pd.read_csv('jobs_in_data.csv')


# ### Filter Fields

# In[3]:


df = dataset[dataset['employee_residence'] == 'United States'].loc[:, ~dataset.columns.isin(['salary_currency', 'salary','job_category'])]
print(df)


# ### Test the Dataset

# In[4]:


jobTypeMean = df.groupby("experience_level")["salary_in_usd"].mean().round(2)
print(jobTypeMean)


# ### Split Dataset into Train and Test Dataframes

# In[11]:


train_df = df.iloc[:7000,:]
test_df = df.iloc[7000:,:]


# ### Covert to TensorFlow Datasets

# In[12]:


label = 'salary_in_usd'
# Will assign each label to an integer value and convert the dataframes to TensorFlow datasets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label, task = tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=label, task = tfdf.keras.Task.REGRESSION)


# ## Train and Fit the Model

# ### Using Random Forest Regression Model

# In[13]:


model = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
model.fit(train_ds)


# ### Model Summary

# In[14]:


model.summary()


# ## Compute Model Accuracy

# In[15]:


model.compile(metrics=['MAPE'])
model.evaluate(train_ds, return_dict=True)
model.evaluate(test_ds, return_dict=True)


# ## Save the Model

# In[16]:


model.save('models/')

