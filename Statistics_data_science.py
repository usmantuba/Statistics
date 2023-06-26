#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime
import scipy.stats


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
#sets the default autosave frequency in seconds
get_ipython().run_line_magic('autosave', '60')
sns.set_style('dark')
sns.set(font_scale=1.2)

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
pd.set_option('display.width', 1000)

np.random.seed(0)
np.set_printoptions(suppress=True)


# In[3]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)
boston_df


# In[5]:


boston_df.info()


# In[6]:


boston_df.describe()


# In[7]:


boston_df.columns


# In[8]:


boston_df.hist(bins=50, figsize=(20,10))
plt.suptitle('Feature Distribution', x=0.5, y=1.02, ha='center', fontsize='large')
plt.tight_layout()
plt.show()


# In[9]:


plt.figure(figsize=(20,20))
plt.suptitle('Pairplots of features', x=0.5, y=1.02, ha='center', fontsize='large')
sns.pairplot(boston_df.sample(250))
plt.show()


# In[10]:


#For the "Median value of owner-occupied homes" provide a boxplot
plt.figure(figsize=(10,5))
sns.boxplot(x=boston_df.MEDV)
plt.title("Boxplot for MEDV")
plt.show()


# In[11]:


#Provide a histogram for the Charles river variable
plt.figure(figsize=(10,5))
sns.distplot(a=boston_df.CHAS,bins=10, kde=False)
plt.title("Histogram for Charles river")
plt.show()


# In[12]:


#Provide a boxplot for the MEDV variable vs the AGE variable. 
#(Discretize the age variable into three groups of 35 years and younger, 
#between 35 and 70 years and 70 years and older)
boston_df.loc[(boston_df["AGE"] <= 35),'age_group'] = '35 years and younger'
boston_df.loc[(boston_df["AGE"] > 35) & (boston_df["AGE"]<70),'age_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df["AGE"] >= 70),'age_group'] = '70 years and older'

boston_df


# In[13]:


plt.figure(figsize=(10,5))
sns.boxplot(x=boston_df.MEDV, y=boston_df.age_group, data=boston_df)
plt.title("Boxplot for the MEDV variable vs the AGE variable")
plt.show()


# In[14]:


#Provide a scatter plot to show the relationship between Nitric oxide concentrations and 
#the proportion of non-retail business acres per town. What can you say about the relationship?
plt.figure(figsize=(10,5))
sns.scatterplot(x=boston_df.NOX, y=boston_df.INDUS, data=boston_df)
plt.title("Relationship between NOX and INDUS")
plt.show()


# In[15]:


#Create a histogram for the pupil to teacher ratio variable
plt.figure(figsize=(10,5))
sns.distplot(a=boston_df.PTRATIO,bins=10, kde=False)
plt.title("Histogram for the pupil to teacher ratio variable")
plt.show()


# In[17]:


# Task 5: Use the appropriate tests to answer the questions provided
boston_df
##Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)
boston_df

## Null Hypothesis(
## ): Both average MEDV are the same

## Alternative Hypothesis(
## ): Both average MEDV are NOT the same
boston_df["CHAS"].value_counts()
a = boston_df[boston_df["CHAS"] == 0]["MEDV"]
a
b = boston_df[boston_df["CHAS"] == 1]["MEDV"]
b
scipy.stats.ttest_ind(a,b,axis=0,equal_var=True)
##Since p-value more than alpha value of 0.05, we failed to reject null hypothesis since there is NO statistical significance.


# In[18]:


##Since p-value more than alpha value of 0.05, we failed to reject null hypothesis since there is NO statistical significance.


# In[21]:


##Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)

boston_df["AGE"].value_counts()
boston_df.loc[(boston_df["AGE"] <= 35),'age_group'] = '35 years and younger'
boston_df.loc[(boston_df["AGE"] > 35) & (boston_df["AGE"]<70),'age_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df["AGE"] >= 70),'age_group'] = '70 years and older'

boston_df

low = boston_df[boston_df["age_group"] == '35 years and younger']["MEDV"]
mid = boston_df[boston_df["age_group"] == 'between 35 and 70 years']["MEDV"]
high = boston_df[boston_df["age_group"] == '70 years and older']["MEDV"]

f_stats, p_value = scipy.stats.f_oneway(low,mid,high)

print("F-Statistic={0}, P-value={1}".format(f_stats,p_value))


# In[22]:



##Since p-value more than alpha value of 0.05, we failed to reject null hypothesis since there is NO statistical significance.


# In[23]:


##Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)


# In[24]:


pearson,p_value = scipy.stats.pearsonr(boston_df["NOX"],boston_df["INDUS"])

print("Pearson Coefficient value={0}, P-value={1}".format(pearson,p_value))


# In[25]:


####Since the p-value (Sig. (2-tailed) < 0.05, we reject the Null hypothesis and conclude that there exists a relationship between Nitric Oxide and non-retail business acres per town.


# In[26]:


##What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)

boston_df.columns


y = boston_df['MEDV']
x = boston_df['DIS']
x = sm.add_constant(x)
results = sm.OLS(y,x).fit()
results.summary()


# In[28]:


np.sqrt(0.062)  ##Pearson Coeffiecent value

##after result##The square root of R-squared is 0.25, which implies weak correlation between both features

##Correlation

boston_df.corr()


plt.figure(figsize=(16,9))
sns.heatmap(boston_df.corr(),cmap="coolwarm",annot=True,fmt='.2f',linewidths=2, cbar=False)
plt.show()


# In[ ]:




