
# coding: utf-8

# 
# # LendingClub Loan Repayment Prediction 
# 
# For this project I am going to explore publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# I will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
# 
# Here are what the columns represent:
# 
# #### credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# #### purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# #### int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# #### installment: The monthly installments owed by the borrower if the loan is funded.
# #### log.annual.inc: The natural log of the self-reported annual income of the borrower.
# #### dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# #### fico: The FICO credit score of the borrower.
# #### days.with.cr.line: The number of days the borrower has had a credit line.
# #### revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# #### revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# #### inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# #### delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# #### pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# # Libraries Import
# 

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Getting the Data
# 
# 

# In[2]:

loans = pd.read_csv('loan_data.csv')


# ** Checking out the csv file usinf different methods.

# In[3]:

loans.columns


# In[4]:

loans.head()


# In[5]:

loans.info()


# In[6]:

loans.describe()


# # Exploratory Data Analysis of Dataset
# 
# Let's do some data visualization! I'll be using seaborn and matplotlib library.
# 
# 

# ** Creating a heatmap of correlation between all the columns of loan dataset

# In[7]:

plt.figure(figsize=(14,7))
sns.heatmap(loans.corr(),annot=True,cmap='inferno',linewidths=1)


# ** Creating a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
# 
# 

# In[8]:

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=45,color='blue',label='C.P - 1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=45,color='red',label='C.P - 0',alpha=0.6)
plt.xlabel("FICO")
plt.legend()


# ** Creating a histogram of two FICO distributions on top of each other, one for each No Fully Paid outcome.**
# 
# 

# In[9]:

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=45,color='blue',label='No Fully paid = 1',alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=45,color='red',label='No Fully paid = 0',alpha=0.6)

plt.xlabel("FICO")
plt.legend()


# ** Creating a countplot using seaborn showing the counts of loans by purpose, with the hue of not.fully.paid. ** 

# In[10]:

plt.figure(figsize=(12,5))
sns.countplot(x='purpose',hue="not.fully.paid",data=loans,palette='Set1')
plt.ylabel("Count of not.fully.paid")


# ** Let's see the trend between FICO score and interest rate. using lmplot **

# In[11]:


sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',height=8)


# # Setting/Preparing up the Data
# 
# Setting up our data for our Random Forest Classification Model!
# 
# **Check loans.info() again.**

# In[12]:

loans.info()


# In[13]:

loans['purpose'].value_counts()


# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies method
# 

# In[14]:

purpose_categories = pd.get_dummies(loans['purpose'],drop_first=True)


# In[15]:

loans_final = pd.concat([loans,purpose_categories],axis=1)


# In[16]:

loans_final.drop(['purpose'],axis=1,inplace=True)


# In[17]:

loans_final.info()


# In[18]:

loans_final.head(3)


# ## Train Test Split
# 
# Now its time to split the data into a training set and a testing set!
# 
# ** Using sklearn to split data into a training set and a testing set.**

# In[19]:

from sklearn.model_selection import train_test_split


# In[20]:

X = loans_final.drop('not.fully.paid',axis=1)
y = loans_final['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)


# ## Training a Decision Tree Model
# 
# 
# ** Import DecisionTreeClassifier**

# In[21]:

from sklearn.tree import DecisionTreeClassifier


# In[22]:

dtc = DecisionTreeClassifier()


# In[23]:

dtc.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# **Creating predictions from the test set and creating a classification report and confusion matrix.**

# In[24]:

predictions = dtc.predict(X_test)


# In[25]:

from sklearn.metrics import classification_report,confusion_matrix


# In[26]:

print("\t\t\tCLASSIFICATION REPORT:\n")
print(classification_report(y_test,predictions))


# In[27]:

print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test,predictions))


# ## Training the Random Forest model
# ** Import RandomForestClassifier**

# In[28]:

from sklearn.ensemble import RandomForestClassifier


# In[29]:

rfc = RandomForestClassifier(n_estimators=999)


# In[30]:

rfc.fit(X_train,y_train)


# ## Predictions and Evaluation of  Random Forest 
# 

# In[31]:

predictions = rfc.predict(X_test)


# **Now creating a classification report from the predictions**

# In[32]:

print("\t\t\tCLASSIFICATION REPORT:\n")
print(classification_report(y_test,predictions))


# **Showing the Confusion Matrix for the predictions.**

# In[33]:

print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test,predictions))


# # Random Forest model prediction is better than Decision Tree.
