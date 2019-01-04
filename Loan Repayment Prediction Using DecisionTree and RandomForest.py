
# coding: utf-8

# 
# # Random Forest Project 
# 
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
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

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Getting the Data
# 
# 

# In[4]:


loans = pd.read_csv('loan_data.csv')


# ** Checking out the csv file usinf different methods.

# In[5]:


loans.columns


# In[6]:


loans.head()


# In[7]:


loans.info()


# In[8]:


loans.describe()


# # Exploratory Data Analysis of Dataset
# 
# Let's do some data visualization! We'll use seaborn and pandas built-in plotting capabilities, but feel free to use whatever library you want. Don't worry about the colors matching, just worry about getting the main idea of the plot.
# 
# 

# ** Creating a heatmap of correlation between all the columns of loan dataset

# In[13]:


plt.figure(figsize=(14,7))
sns.heatmap(loans.corr(),annot=True,cmap='inferno',linewidths=1)


# ** Creating a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
# 
# 

# In[17]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=45,color='blue',label='C.P - 1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=45,color='red',label='C.P - 0',alpha=0.6)
plt.xlabel("FICO")
plt.legend()


# ** Creating a histogram of two FICO distributions on top of each other, one for each No Fully Paid outcome.**
# 
# 

# In[18]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=45,color='blue',label='No Fully paid = 1',alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=45,color='red',label='No Fully paid = 0',alpha=0.6)

plt.xlabel("FICO")
plt.legend()


# ** Creating a countplot using seaborn showing the counts of loans by purpose, with the hue of not.fully.paid. ** 

# In[19]:


plt.figure(figsize=(12,5))
sns.countplot(x='purpose',hue="not.fully.paid",data=loans,palette='Set1')
plt.ylabel("Count of not.fully.paid")


# ** Let's see the trend between FICO score and interest rate. using lmplot **

# In[21]:



sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',height=8)


# # Setting/Preparing up the Data
# 
# Let's get ready to set up our data for our Random Forest Classification Model!
# 
# **Check loans.info() again.**

# In[23]:


loans.info()


# In[24]:


loans['purpose'].value_counts()


# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies method
# 

# In[36]:


purpose_categories = pd.get_dummies(loans['purpose'],drop_first=True)


# In[37]:


loans_final = pd.concat([loans,purpose_categories],axis=1)


# In[43]:


loans_final.drop(['purpose'],axis=1,inplace=True)


# In[44]:


loans_final.info()


# In[45]:


loans_final.head(3)


# ## Train Test Split
# 
# Now its time to split our data into a training set and a testing set!
# 
# ** Use sklearn to split your data into a training set and a testing set as we've done in the past.**

# In[47]:


from sklearn.model_selection import train_test_split


# In[96]:


X = loans_final.drop('not.fully.paid',axis=1)
y = loans_final['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)


# ## Training a Decision Tree Model
# 
# 
# 
# ** Import DecisionTreeClassifier**

# In[97]:


from sklearn.tree import DecisionTreeClassifier


# In[98]:


dtc = DecisionTreeClassifier()


# In[99]:


dtc.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# **Creating predictions from the test set and creating a classification report and confusion matrix.**

# In[100]:


predictions = dtc.predict(X_test)


# In[101]:


from sklearn.metrics import classification_report,confusion_matrix


# In[102]:


print("\t\t\tCLASSIFICATION REPORT:\n")
print(classification_report(y_test,predictions))


# In[103]:


print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test,predictions))


# ## Training the Random Forest model
# ** Import RandomForestClassifier**

# In[83]:


from sklearn.ensemble import RandomForestClassifier


# In[84]:


rfc = RandomForestClassifier(n_estimators=999)


# In[85]:


rfc.fit(X_train,y_train)


# ## Predictions and Evaluation of  Random Forest 
# 

# In[86]:


predictions = rfc.predict(X_test)


# **Now creating a classification report from the predictions**

# In[87]:


print("\t\t\tCLASSIFICATION REPORT:\n")
print(classification_report(y_test,predictions))


# **Showing the Confusion Matrix for the predictions.**

# In[88]:


print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test,predictions))


# # Random Forest model prediction is better than Decision Tree
