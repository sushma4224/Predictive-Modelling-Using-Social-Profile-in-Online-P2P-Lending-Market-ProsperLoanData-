#!/usr/bin/env python
# coding: utf-8

# #   Predictive Modelling Using Social Profile in Online P2P Lending Market

# # Prosper Loandata
# 

# # Problem Statement :

# Online peer-to-peer (P2P) lending markets enable individual consumers to borrow from, and lend money to, one another directly. We study the borrower-,loan- and group- related determinants of performance predictability in an online P2P lending market by conceptualizing financial and social strength to predict borrower rate and whether the loan would be timely paid.

# Importing all necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings ('ignore')


# Loading the Data set: We have 113937 rows × 81 columns

# In[2]:


df= pd.read_csv("C:/Users/murthy/Downloads/prosperLoanData.csv")


# In[3]:


df


# In[4]:


df.head()


# Statistical Information

# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.describe()


# Checking Data type of Attributes

# In[8]:


df.info()


# Checking Unique values in dataset :

# In[9]:


df.apply(lambda x: len(x.unique()))


# # Preprocessing the Dataset

# Checking duplicate values:

# In[10]:


df.duplicated().sum()


# We observe that dataset contains no duplicate values.

# Checking Missing (null) values:

# In[11]:


df.isnull().sum()


# We observe many attributes have missing values.

# In[12]:


print(df.isnull().values.sum())


# We have Categorical as well as numerical attributes which we will process seperately.

# In[13]:


categorical=df.select_dtypes("object")
categorical


# In[14]:


continuous=df.select_dtypes("number")
continuous


# Checking Missing (null) values of Categorical attributes:

# In[15]:


categorical.isna().sum()


# Filling Missing values of categorical attributes using Mode funcion:

# In[16]:


categorical=categorical.fillna(categorical.mode().iloc[0])
categorical


# In[17]:


categorical.isna().sum()


# All the missing values of categorical attributes are now filled.

# Checking Missing (null) values of Numerical attributes:

# In[18]:


continuous.isna().sum()


# Filling Missing values of Numerical attributes using Median funcion:

# In[19]:


continuous=continuous.fillna(continuous.median().iloc[0])
continuous


# In[20]:


continuous.isna().sum()


# All the missing values of Numerical attributes are now filled.

# Combining the Categorical continuous data into one file named "ndf" using concat function with the inner join condition.

# In[21]:


ndf=pd.concat([categorical,continuous],axis=1,join='inner')
ndf


# Checking again wether there is still have missing values or not

# In[22]:


ndf.isna().sum()


# Displaying column names

# In[23]:


ndf.columns


# In[24]:


#ndf['LoanOriginationDate']=pd.to_datetime(ndf['LoanOriginationDate'])


# # Data Cleaning:

# Removing unnecessary columns from the dataset

# In[25]:


ndf.drop(['ListingCreationDate','LoanOriginationDate','GroupKey','CreditGrade','ProsperPrincipalBorrowed','ProsperPrincipalOutstanding','EstimatedEffectiveYield','EstimatedLoss','EstimatedReturn','ProsperRating (numeric)','TotalProsperLoans','TotalProsperPaymentsBilled','OnTimeProsperPayments','ProsperPaymentsLessThanOneMonthLate','ProsperPaymentsOneMonthPlusLate','ListingKey'],axis=1,inplace=True)


# In[26]:


ndf.head()


# # Label Encoding:

# Label Encoding is to convert the categorical column into the numerical column.

# In[27]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cols=['ClosedDate','ProsperRating (Alpha)','BorrowerState','Occupation','EmploymentStatus','FirstRecordedCreditLine','DateCreditPulled','IncomeRange','LoanKey','LoanOriginationQuarter','MemberKey']
ndf[cols]=ndf[cols].apply(LabelEncoder().fit_transform)
ndf.head()


# One Hot Encoding:

# we can also use one hot encoding for categorical columns.

# In[28]:


Ls=pd.get_dummies(ndf['LoanStatus'])
print(Ls)


# It will create a new column for each category. Hence, it will add the corresponding category instead of numerical values. if the corresponding location type i present it will show as "1" orelse it will show "0"

# In[29]:


Ndf=pd.concat((Ls,ndf),axis=1)
Ndf=Ndf.drop(['LoanStatus'],axis=1)
Ndf=Ndf.drop(['Cancelled','Chargedoff','Current','Defaulted','FinalPaymentInProgress','FinalPaymentInProgress','Past Due (1-15 days)','Past Due (16-30 days)','Past Due (31-60 days)','Past Due (61-90 days)','Past Due (91-120 days)','Past Due (>120 days)'],axis=1)
Newdf=Ndf.rename(columns={"Completed":"LoanStatus"})
print(Newdf)


# Here Loan Status 1 = Completed and 0 = Not Completed

# In[30]:


Newdf


# In[31]:


Newdf.columns


# # Exploratory Data Analysis

# ### Univariate Analysis

# Original Loan Amount for loans taken

# In[32]:


plt.figure(figsize=[20,8])
plt.hist(data=df,x="LoanOriginalAmount")
plt.title("Loan Original Amount Distribution")
plt.xlabel("Loans")
plt.ylabel("count of Loans")


# Majority of the original loan amounts are below $20,000.

# Loan interest ans fees for loan taken

# In[33]:


plt.figure(figsize=[20,8])
plt.hist(data=df,x="LP_InterestandFees")
plt.title("Loan Interesta and Fees Distribution")
plt.xlabel("Interst and Fees")
plt.ylabel("count of Loans")


# Majority of Interst and Fees charged for loans are below $4,000.

# Most and last count of loans given years

# In[34]:


#chaging Loan Origination Date into Date Time format
df['LoanOriginationDate']= pd.to_datetime(df['LoanOriginationDate'])


# In[35]:


plt.figure(figsize=[8,5])
base_color=sns.color_palette()[0]
sns.countplot(data=df,x=df['LoanOriginationDate'].dt.year,color=base_color)
plt.title('Loan Originated by Year Distribution')
plt.xlabel('Year Loan was Given')


# 2013 had the highest number of loans given  while 2005 had the least.

# Loan Terms for Loans given

# In[36]:


plt.figure(figsize=[25,10])
base_color=sns.color_palette()[0]
sns.countplot(data=df,x=df['Term'],color=base_color)
plt.title('Loan Terms in Months')
plt.xlabel('Terms in Month')


# Majority of the loans had given 36-month term and 12-month terms had the least number of loans given.

# Employment Status for the borrower

# In[37]:


plt.figure(figsize=[10,8])
base_color=sns.color_palette()[0]
cate_order=df.EmploymentStatus.value_counts().index
sns.countplot(data=df,x=df['EmploymentStatus'],color=base_color,order=cate_order)
plt.title('Borrowers-Employment Distribution')
plt.ylabel('Terms in months')
plt.xlabel('Employment Status')


# Majority of the borrower were employed while the retires took the least number of loans.

# Status for all the loans given are

# In[38]:


plt.figure(figsize=[10,8])
base_color=sns.color_palette()[0]
cate_order=df.LoanStatus.value_counts().index
sns.countplot(data=df,y=df['LoanStatus'],color=base_color,order=cate_order)
plt.title('Status of Loan Distribution')
plt.ylabel('Loan Status')


# Majority of the loans are current.

# Income Range for the Borrowers

# In[39]:


plt.figure(figsize=[15,10])
base_color=sns.color_palette()[0]
cate_order=df.IncomeRange.value_counts().index
sns.countplot(data=df,x=df['IncomeRange'],color=base_color,order=cate_order)
plt.title('Borrowers-IncomeRange Distribution')
plt.xlabel('IncomeRange')


# Borrowers with an income range of between $25,000 to 49,999 were the majority followed closely by thosein the bracket of $50,000 to 74,999. Those categorized in the $0 were the least.

# Highest number of BorrowerRate

# In[40]:


plt.figure(figsize=[10,8])
plt.hist(df['BorrowerRate'])
plt.title('Borrower Rate Distribution')
plt.xlabel('Borrower Rate')
plt.ylabel('Count of Loans')


# The highest number of Borrower Rat is between 0.1 and 0.2

# Borrower is a homeowner?

# In[41]:


plt.figure(figsize=[25,10])
base_color=sns.color_palette()[0]
cate_order=df.IsBorrowerHomeowner.value_counts().index
sns.countplot(data=df,x='IsBorrowerHomeowner',color=base_color,order=cate_order)
plt.title('Borrowers-Homeowner Status Distribution')
plt.xlabel('Borrower homeownership Status')


# Majority of the Borrowers own a home. The difference between the two is negligable by less than 1% though.

# Borrowers Occupations

# In[42]:


plt.figure(figsize=[20,20])
base_color=sns.color_palette()[0]
cate_order=df.Occupation.value_counts().index
sns.countplot(data=df,y=df['Occupation'],color=base_color,order=cate_order)
plt.title(' Occupations of the Borrowers Distribution')


# Students formed the group of the least loaned while others and professionals seemed to form the group of the most loaned.

# ### Bivariate Analysis

# Relationship between Employment Status and Loan Amount Given

# In[43]:


plt.figure(figsize=[18,12])
base_color=sns.color_palette()[0]
cate_order=df.EmploymentStatus.value_counts().index
sns.boxplot(data=df,x=df['LoanOriginalAmount'],y=df['EmploymentStatus'],color=base_color,order=cate_order)
plt.title('Employment Status and Loan Original Amount Distribution')
plt.xlabel('LoanOriginalAmount')
plt.ylabel('Employment Status')


# Being employed gives a borrower a chance to access more loans.

# Relationship between Credit Grade and Monthly Loan Payment

# In[44]:


df['CreditGrade'] = df['CreditGrade'].fillna('No Grade')


# In[45]:


plt.figure(figsize=[18,10])
base_color=sns.color_palette()[0]
sns.boxplot(data=df,x=df['MonthlyLoanPayment'],y=df['CreditGrade'],color=base_color)
plt.title('Credit Grade and Monthly Loan Payment Distribution')
plt.xlabel('Monthly Loan Payment')
plt.ylabel('Credit Grade')


# Loans that fell in the Credit-Grade groups of HR, E and NC had the lowest monthly loan payments on average while those in A and AA paid more.

# ### Multivariate Analysis

# Relationship between Monthly Loan Payment, Loan Status and Prosper Score

# In[46]:


plt.figure(figsize=[18,10])
plt.scatter(data=df,x='MonthlyLoanPayment', y='LoanStatus',c='ProsperScore')
plt.colorbar(label="ProsperScore")
plt.title('MonthlyLoanPayment and LoanStatus vs ProsperScore')
plt.xlabel('MonthlyLoanPayment')
plt.ylabel('LoanStatus')


# Completed and Current Loans seemed to have the lowest risks while the other which were mostly past due had higher risks.

# Relationship between Borrower Rate , Borrower APR and ProsperScore

# In[47]:


plt.figure(figsize=[18,10])
plt.scatter(data=df,x='BorrowerRate', y='BorrowerAPR',c='ProsperScore')
plt.colorbar(label="ProsperScore")
plt.title('BorrowerRate and Borrower APR vs ProsperScore')
plt.xlabel('BorrowerRate')
plt.ylabel('Borrower APR')


# BorrowerAPR and BorrowerRate have a strong positive relatioship as seen above, an increase or decrease in either affects the other directly. Majority of loans with lower BorrowerAPR and BorrowerRate have higher ProsperScore(lower risks) and visa-versa.

# # Feature Engineering

# ### Splitting data into Training and Testing data

# In[48]:


X= Newdf.drop(columns="LoanStatus", axis=1) 
Y= Newdf["LoanStatus"]


# In[49]:


X


# In[50]:


Y


# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.20 , random_state = 100)


# ### Mutual Information Selection

# In[52]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train , Y_train)
mutual_info


# In[53]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending = False)


# In[54]:


#selecting the important features
from sklearn.feature_selection import SelectKBest
imp_features =SelectKBest(mutual_info_classif,k=10)
imp_features.fit(X_train , Y_train)
X_train.columns[imp_features.get_support()]


# In[55]:


X_Train = imp_features.transform(X_train)
X_Test = imp_features.transform(X_test)


# ### Standardizing the Data

# In[56]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform (X_Test)


# ### Princial Component Analysis

# In[57]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_Train = pca.fit_transform(X_Train)
X_Test = pca.fit_transform(X_Test)


# In[58]:


variance=pca.explained_variance_ratio_


# In[59]:


variance


# ### Model Fitting

# In[60]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_Train,Y_train)


# In[61]:


Y_pred=classifier.predict(X_Test)


# In[65]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[67]:


cm=confusion_matrix(Y_test,Y_pred)
cm


# In[68]:


(13223+4885)/(13223+1956+2724+4885)


# In[66]:


accuracy_score(Y_test,Y_pred)


# 79% Accuracy

# In[ ]:




