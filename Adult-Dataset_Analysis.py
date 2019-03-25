
# coding: utf-8

# # Exploratory Data Analysis - Adult Dataset 

# In[1]:


# Data exploration libraries
import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import randint as sp_randint

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns 

# Sklearn package import
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Misc libraries
import pickle
import json
from sklearn.pipeline import make_pipeline


# In[2]:


# Data load
header_list=['age','workclass','fnlwgt','education','education-num','marital-status',
             'occupation','relationship','race','sex',
             'capital-gain','capital-loss','hours-per-week','native-country','income_class']
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            names=header_list,index_col=False)

data['income_class']=data['income_class'].astype('str')

data['target']=np.where(data['income_class']==data.income_class[1] , 0,1)


# In[3]:


data.head()


# In[4]:


# Data type and size 
data.info()


# In[5]:


##### There are 32651 entries/rows for each 16 columns. The Target column contains the binary class of people above or 
##### below $50k. Our analysis is based on evaluating given other parameters what is the Income class for the person.


# In[6]:


from IPython.display import display_html

def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
  

column=[ 'workclass', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'native-country',
       'income_class', 'target']
  
df1=pd.DataFrame({'Occupation':data.occupation.unique()})

df2=pd.DataFrame({'Workclass':data.workclass.unique()})

education_df=pd.DataFrame({'Education_Label':data.education.unique(), 'Education_Number':data['education-num'].unique()})

df4=pd.DataFrame({'Relationship':data.relationship.unique()})

df5=pd.DataFrame({'Race':data.race.unique()})

df6=pd.DataFrame({'Income_Class':data.income_class.unique()})

print('The categories for each column are as follows :  ')
display_side_by_side(df1,df2,education_df.sort_values(by='Education_Number'), df4,df5, df6)


# In[7]:


# checking the education related information 

print(data.education.unique())
print(data['education-num'].unique())

education_df=pd.DataFrame({'Education_Label':data.education.unique(), 'Education_Number':data['education-num'].unique()})
print('\n The Numeric values assigned to each education level : ')
education_df.sort_values(by='Education_Number')


# In[8]:


##### I notice that each education level is asigned a corresponsing numeric value which is increasing order of level of 
##### education attained, For example the Pre school is considered as basic level of education with numeric value 1
##### and Doctorate being highest level of educaiton attained is given number 16


# ### Check how much data we have for each category in the dataset  

# In[9]:


# Visualization of the available data 
fig = plt.figure(figsize=(15, 15))
plt.rc('font', size=11)
plt.rc('axes', axisbelow=True)

list_1=['education','marital-status','workclass','occupation','race','age','sex','income_class','relationship']

for i in list_1:  
    
    if i=='age':
        sub1 = plt.subplot(3, 3, list_1.index(i)+1)
        data['age'].plot(kind='hist', bins=20,edgecolor='black',color='green',alpha=0.8)
        plt.title('Entries for age groups ')
        plt.xlabel('Age')
        plt.tight_layout()
        plt.grid()
    else :    
        sub1 = plt.subplot(3, 3, list_1.index(i)+1)
        data[i].value_counts().plot(kind='bar',color='green',alpha=0.8)
        plt.title('Counts of each level of '+ i)
        plt.grid()
plt.savefig('Data_available_1.png', dpi=600, bbox_inches='tight')
plt.tight_layout()

plt.figure(figsize=(40, 13))
sub1 = plt.subplot(3, 3, 8)
plt.rc('font', size=12)
data['native-country'].value_counts().plot(kind='bar',alpha=0.8, color='green')
plt.title('Counts of each native country')
plt.grid()
plt.tight_layout()
plt.savefig('Data_available.png', dpi=600, bbox_inches='tight')
plt.show()


# # Figure A (above)

# In[10]:


(data==' ?').any()


# > I notice some '?' in workclass and native country and occupation level. We will remove them eventually. 

# In[11]:


remove=['native-country','occupation','workclass']
for i in remove :
    data.drop(data.loc[data[i]==' ?'].index,inplace=True)


# In[12]:


data.info()


# In[13]:


##### We have lost around 2399 rows. '?' were mostly in the occupation column 
##### but i would not be guessing some random occupation to fill in the '?' hence i am okay with removing these entries.


# In[14]:


##### The target variable contains around 22600 entries for the category of people earning <$50k and around 
##### 7500 entries of people earning moer than $50k. This is important observation indicating that our datset is biased
##### towards people earning less than $50k.


# ### How many people are above 50k range in each category 

# In[15]:


categories= ['education', 'workclass', 'marital-status', 'occupation', 'relationship','sex','race']

fig = plt.figure(figsize=(15, 15))
plt.rc('font', size=11)

for cat in categories:
    sub1 = plt.subplot(3, 3, categories.index(cat)+1)    
    data.groupby(cat).mean()['target'].plot(kind='bar',color='green',alpha=0.8)
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.tight_layout()
    plt.xlabel('')
    plt.ylabel('Proportion')
    plt.title('Proportion with >50k income per '+ cat)
plt.savefig('Analysis_Income_prediction_1.png', dpi=600, bbox_inches='tight')   
    
plt.rc('font', size=12)
plt.figure(figsize=(40, 15))
plt.rc('font', size=25)
data.groupby('native-country').mean()['target'].plot(kind='bar',color='green',alpha=0.8)
plt.rc('font', size=12)
plt.grid()
plt.tight_layout()
plt.savefig('Analysis_Income_prediction.png', dpi=600, bbox_inches='tight')
plt.show()


# # Figure B (above)

# In[16]:


# Education 
##### The wealthy people are generally highly educated. Professors, Doctorate and or masters education 
##### level people earn well
# Lets investigate who among low education levels are wealthy what they do how come they can earn so much at such young age 


# In[17]:


fig = plt.figure(figsize=(15, 5))

data[np.logical_and(data['education-num']<=8,
                    data['target']==1)].groupby('workclass').count()['age'].plot(kind='bar',
                                                                                 color='green',alpha=0.8)
plt.title('Wealthy with education less than 12th Standard based on work profile ')
plt.ylabel('Number of people >$50k')
plt.grid()
plt.savefig('Analysis_Income_prediction_work_profile.png', dpi=600, bbox_inches='tight')


# In[18]:


##### Most of the people with education level less than 12th standard work in Private jobs 


# In[19]:


# Work class
##### The Self employed people have a higher proportion of being rich (>50k $) followed by people working in Federal jobs


# In[20]:


# Marital Status 
##### The Married couple from Armed forces and Civilian spouse are in high income category but the dataset 
##### containes very few entries for Armed force category hence we wont consider them as much of a valid observation 


# In[21]:


# Occupation 
##### The Executive and Managerial roles are the most paid ones it seems followed by  Professors and Protective services 
##### Some of the job categories such as Clerical jobs, farming fishing and Cleaners and handlers are not that much paid.


# In[22]:


# Relation ship 
##### Its a little surprising for me to see Husbands having less proportion than Wives with high income, 
##### I notice that the data for wives is just ~1400  entries and for the husbands is ~ 12500 
##### because of which the proportion is a little misleading 


# In[23]:


# Gender 
##### The proportion of males with high income is more than the females 


# In[24]:


# Native Country 
##### As we saw above the data for United states natives is overwhelmingly higher than other countries. 
##### The proportion of people who got wealthy (>$50k) from different natives is very high for France, Taiwan, Iran. Again its worth noting
##### that the data for each of these countries is too less to make a sane judgement. 


# ### Checking Distribution of income among different age groups 

# In[25]:


fig = plt.figure(figsize=(15, 6))
plt.legend=['<50k','>=50k']
data['target']=data['target'].astype('category')
ax=sns.boxplot(y='target',x='age',data=data.loc[:,['age','target']],showbox=False)
ax=sns.violinplot(y='target',x='age',data=data.loc[:,['age','target']])

ax.set(yticklabels=['<=50k($)','>50k($)'])
plt.ylabel('Income groups')
plt.setp(ax.collections, alpha=.3)
plt.title('Distribution of income among different Age groups ')
plt.savefig('Analysis_Income_predictionAge.png', dpi=600, bbox_inches='tight')
plt.show()


# In[26]:


# Income vs Age 
##### Generally people between the age group of 30-50 are wealthy. The youngsters upto the age of 27 are under 
##### the low income category. This makes sense as this is the age when the students are studying or just getting in 
##### to the employment 


# In[27]:


fig = plt.figure(figsize=(40, 20))
ax=sns.catplot(y="target", x="hours-per-week", data=data.loc[:,['hours-per-week','target']], alpha=0.1);
plt.title('Work hours per week vs Income group ')
plt.ylabel('Income Group')
plt.grid()
#plt.savefig('Analysis_Income_prediction_workhours.png', dpi=100, width=20, height=10)
ax.savefig("Analysis_Income_prediction_workhours.png")


# In[28]:


# Hours per week
##### The people in the higher income group work mostly between 35-60 hours a week. 
##### This goes up to 100 as well but there are less of such peopl.


# In[29]:


# Race 
##### We notice that we have too little data for races other than White(Figure A). Still i tried to compare compare the 
##### proportions of each race who were wealthy (>50k$). For the whites ~ 26% people are earning >50k while from the available 
##### data ~28% Asian Pac Islander earn greater than $50k


# In[30]:


data_gain_loss=data.loc[:,['capital-gain',
                           'capital-loss','target']][np.logical_or(data['capital-gain']!=0,
                                                                                  data['capital-loss']!=0)]
data_gain_loss.head()


# # Model fitting data preparation  

# In[31]:


# Converting the object variable types to integer for further analysis


# In[32]:


header_list=['age','workclass','fnlwgt','education','education-num','marital-status',
             'occupation','relationship','race','sex',
             'capital-gain','capital-loss','hours-per-week','native-country','income_class']
train_data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            names=header_list,index_col=False)

train_data['income_class']=train_data['income_class'].astype('str')

print('Shape of Train dataset is : ', train_data.shape)


# In[33]:


test_data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
            names=header_list,index_col=False, skiprows=[0])

test_data.head(2)
print('Shape of Test dataset is : ',test_data.shape)


# In[34]:


test_data.income_class.unique()


# In[35]:


data=pd.concat([train_data,test_data])


# In[36]:


# Assigning Male and Females : 1 and 0 Integer Values
data['sex']=data['sex'].map({' Male':1,' Female':0}).astype(int)


# In[37]:


# Data set is skewed towards White race hence two categories look ok. 1 for White and 0 for the rest.
data['race']=data['race'].map({' White':1,' Black':0, ' Asian-Pac-Islander':0, ' Amer-Indian-Eskimo':0,
       ' Other':0}).astype(int)


# In[38]:


# The income range looks distinct for students upto standard 12th education, then Associates and then people with 
# Bachelors degree or above, Hence I am dividing this category into 3 classes

data['education']=data['education'].replace([ ' 11th', ' 9th',
       ' 7th-8th', ' 5th-6th', ' 10th', ' Preschool',
       ' 12th', ' 1st-4th'],'Lower_Edu')

data['education']=data['education'].replace([' HS-grad', ' Some-college', ' Assoc-acdm', ' Assoc-voc'],'Middle_Edu')


data['education']=data['education'].replace([' Bachelors', ' Masters', ' Doctorate',
        ' Prof-school',],'Higher_Edu')

data['education']=data['education'].map({'Lower_Edu':0,'Middle_Edu':1,'Higher_Edu':2}).astype(int)
    


# In[39]:


# All the native United States belong to category 1 and the rest belong to 0
data['native-country']=np.where(data['native-country']==' United-States', 1, 0).astype(int)


# In[40]:


# The people working in government belong to category 2, those involved in Private jobs have category 0 and rest of them
# working for themselves are in category 1

data["workclass"] = data["workclass"].replace([' State-gov', ' Federal-gov', 
                                                 ' Local-gov'], 'Gov')

data["workclass"] = data["workclass"].replace([' Self-emp-not-inc', ' Self-emp-inc', 
                                                 ' Without-pay', ' Never-worked'], 'Self')

data["workclass"] = data["workclass"].map({" Private":0, "Self":1, "Gov":2, ' ?':-1}).astype(int)


# In[41]:


# Assigning the categories to the occupations. Some occupations are believed to be earning high so i assigned 
# them high pay category. Similarly Middle and Lower Pay categories.

data["occupation"] = data["occupation"].replace([' Exec-managerial', ' Prof-specialty', 
                                                 ' Protective-serv',' Tech-support' ], 'HighPay')

data["occupation"] = data["occupation"].replace([' Craft-repair', ' Sales', 
                                                 ' Transport-moving'], 'MiddlePay')

data["occupation"] = data["occupation"].replace([' Priv-house-serv', ' Farming-fishing', 
                                                 ' Armed-Forces',' Machine-op-inspct',
                                                ' Other-service',' Handlers-cleaners', ' Adm-clerical'], 'LowPay')



data["occupation"] = data["occupation"].map({"LowPay":0, "MiddlePay":1, "HighPay":2, ' ?':-1}).astype(int)


# In[42]:


# For the marital status i have assigned three categories. The unmarried people. The people in marriage and 
# another category for those who have separated due to some reason.

data["marital-status"] = data["marital-status"].replace([' Never-married' ], 'Single')

data["marital-status"] = data["marital-status"].replace([' Married-civ-spouse', ' Married-AF-spouse'], 'Couple')

data["marital-status"] = data["marital-status"].replace([' Divorced', ' Married-spouse-absent', 
                                                 ' Separated',
                                                ' Widowed'], 'Separated')

data["marital-status"] = data["marital-status"].map({"Single":0, "Couple":1, "Separated":2}).astype(int)


# In[43]:


data.shape


# In[44]:



data['income_class']=data['income_class'].replace([' <=50K', ' <=50K.'],'<=50K')
data['income_class']=data['income_class'].replace([' >50K',  ' >50K.'],'>50K')

data["income_class"] = data["income_class"].map({'>50K':1, '<=50K':0}).astype(int)


# In[45]:


data.head()


# In[46]:


data=data.drop(['fnlwgt', 'relationship','education-num'], axis=1);


# In[47]:


data.head()


# In[48]:


# I will use this train data for the model building
train_dataset=data.iloc[0:32561,]


# In[49]:


# I Will keep this test dataset aside 
test_dataset=data.iloc[32561:48842,]


# In[50]:


##### For doing the further analysis i am thinking of removing the predictors which are not very insightful.
##### Capital-gain and Capital-Loss are not adding much value as large number of observations have 0 capital gain and capital loss.
##### Native country as well has entries mostly for United States and is not adding much value to my ML model.
##### fnlwgt is the weight which again could not bring much of an insight and hence i am removing that column as well.
##### We alread have education levels in numbers with us which are very important for the analysis but the names of these education levels
##### are redundant for model fitting atleast hence i am removing the education column.


# In[51]:


X=train_dataset.drop(['income_class'],axis=1)
y=train_dataset['income_class']


# In[52]:


# Split the remaining data in 80:20 ratio for train and validation datasets
np.random.seed(123)
X_train,X_valid,y_train, y_valid = train_test_split(X,y,test_size=0.2)


# In[53]:


##### Since its a binary classification problem I will use the following models to predict the classes. 
##### 
##### 1 - Logistic regresion
##### 2 - SVM
##### 3 - Decision Trees / Random Forest
##### 4 - Neural network (Non parametric)
##### 5 - knn 
##### 
##### I will fit the models on the train data and will be using the Validation dataset in the process. 
##### After the final model is chosen, I will check the performance on Test Data.


# In[54]:


# Creating X_test and y_test for final check on the model 
X_test=test_dataset.drop(['income_class'],axis=1)
y_test=test_dataset['income_class']


# ### Random Forest Classifier

# In[55]:


model=RandomForestClassifier()

model.fit(X_train,y_train)
print('The train score is : ', "{00:.2f}%".format(round(model.score(X_train, y_train),4)*100))
print('The Validation score is : ',"{00:.2f}%".format(round(model.score(X_valid, y_valid),4)*100))
print('The Test score is : ',"{00:.2f}%".format(round(model.score(X_test, y_test),4)*100))


# In[56]:


# Applying Randomized search to find the optimum parameters 

param_dist = dict({'max_depth' : np.arange(1,30), 'max_features': np.arange(1,12)})

model_rf=RandomForestClassifier(n_estimators=30)

model_grid=RandomizedSearchCV(model_rf,param_dist,cv=10, n_jobs=-1, n_iter=20, random_state=123)
model_grid.fit(X_train,y_train)

print('The Best Features for Random Forest Are : ',model_grid.best_params_)


# In[57]:


model_best=RandomForestClassifier(max_features=8, max_depth=11, random_state=123)

model_best.fit(X_train,y_train)
print('The train score is : ', "{00:.2f}%".format(round(model_best.score(X_train, y_train),4)*100))
print('The Validation score is : ',"{00:.2f}%".format(round(model_best.score(X_valid, y_valid),4)*100))
print('The Test score is : ',"{00:.2f}%".format(round(model_best.score(X_test, y_test),4)*100))

train_acc_rf="{00:.2f}%".format(round(model_best.score(X_train, y_train),4)*100)
valid_acc_rf="{00:.2f}%".format(round(model_best.score(X_valid, y_valid),4)*100)
test_acc_rf="{00:.2f}%".format(round(model_best.score(X_test, y_test),4)*100)


# In[58]:


# There is improvement in the Test Accuracy from 84.12% to 86.25% which is good sign that our hyper parameter selection 
# from RandomSearch did add value 


# ### Logistic regression accuracy 

# In[59]:


model_lr=LogisticRegression()
model_lr.fit(X_train,y_train)


# In[60]:


print('The train score is : ', "{00:.2f}%".format(round(model_lr.score(X_train, y_train),4)*100))
print('The Validation score is : ',"{00:.2f}%".format(round(model_lr.score(X_valid, y_valid),4)*100))
print('The Test score is : ',"{00:.2f}%".format(round(model_lr.score(X_test, y_test),4)*100))


# In[61]:


param_dist = dict({'C' : np.logspace(-3,3,7), "penalty":["l1","l2"]})

model_lr=LogisticRegression()

model_grid_lr=GridSearchCV(model_lr,param_dist,cv=10, n_jobs=-1)
model_grid_lr.fit(X_train,y_train)

print('The Best Features for Logistic Regression are : ',model_grid_lr.best_params_)


# In[62]:


model_lr_best=LogisticRegression(C=10.0, penalty='l1')
model_lr_best.fit(X_train,y_train)

print('The train score is : ', "{00:.2f}%".format(round(model_lr_best.score(X_train, y_train),4)*100))
print('The Validation score is : ',"{00:.2f}%".format(round(model_lr_best.score(X_valid, y_valid),4)*100))
print('The Test score is : ',"{00:.2f}%".format(round(model_lr_best.score(X_test, y_test),4)*100))
train_acc_lr="{00:.2f}%".format(round(model_lr_best.score(X_train, y_train),4)*100)
valid_acc_lr="{00:.2f}%".format(round(model_lr_best.score(X_valid, y_valid),4)*100)
test_acc_lr="{00:.2f}%".format(round(model_lr_best.score(X_test, y_test),4)*100)


# ### XG Boost 

# In[63]:


model_xgb=XGBClassifier(n_estimators=30,booster='gbtree')
parameters_xgb=dict({'max_depth':np.arange(1,30), 'learning_rate':np.arange(0,1,0.01)})

model_xgb_rs=RandomizedSearchCV(model_xgb,parameters_xgb,cv=5,n_iter=20,n_jobs=-1, random_state=123)


# In[64]:


model_xgb_rs.fit(X_train,y_train)

print('The best parameters for XG Boost are : ',model_xgb_rs.best_params_ )


# In[65]:


model_xgb_best=XGBClassifier(learning_rate=0.85, max_depth=4, n_estimators=30, booster='gbtree', random_state=123)
model_xgb_best.fit(X_train,y_train)
print('The train score is : ', "{00:.2f}%".format(round(model_xgb_best.score(X_train, y_train),4)*100))
print('The Validation score is : ',"{00:.2f}%".format(round(model_xgb_best.score(X_valid, y_valid),4)*100))
print('The Test score is : ',"{00:.2f}%".format(round(model_xgb_best.score(X_test, y_test),4)*100))

train_acc_xgb="{00:.2f}%".format(round(model_xgb_best.score(X_train, y_train),4)*100)
valid_acc_xgb="{00:.2f}%".format(round(model_xgb_best.score(X_valid, y_valid),4)*100)
test_acc_xgb="{00:.2f}%".format(round(model_xgb_best.score(X_test, y_test),4)*100)


# In[66]:


# Final Model Scores 


# In[67]:


pd.DataFrame({'Model':['Random Forest','Logistic Regression','XGBoost'], 
             'Train Accuracy':[train_acc_rf,train_acc_lr,train_acc_xgb], 
              'Test Accuracy':[test_acc_rf,test_acc_lr,test_acc_xgb],
              'Validation Accuracy':[valid_acc_rf,valid_acc_lr,valid_acc_xgb]})


# In[68]:


# Since the XGBoost seems to be performing the best but it is trouble some to put it on the Heroki platform hence for now 
# I will go with the Randome Forest which is doing good as well.


# ### Combining the datasets from Train Test and Validation to train the model on the entire data set

# In[69]:


print('The shape of X_train ',X_train.shape)
print('The shape of X_test ',X_test.shape)
print('The shape of X_validation ',X_valid.shape)


# In[70]:


Entire_X=pd.concat([X_train,X_valid,X_test])
Entire_y=pd.concat([y_train,y_valid,y_test])


# In[71]:


# Training the Random Forest model on the Entire Dataset

model_best=RandomForestClassifier(max_features=8, max_depth=11, random_state=123)
model_best.fit(Entire_X,Entire_y)


# In[72]:


with open('columns.json', 'w') as fh:
    json.dump(Entire_X.columns.tolist(), fh)


# In[73]:


with open('dtypes.pickle', 'wb') as fh:
    pickle.dump(Entire_X.dtypes, fh)


# In[74]:


from sklearn.externals import joblib
joblib.dump(model_best, 'model.pickle')


# In[75]:


with open('columns.json', 'r') as fh:
    columns = json.load(fh)


# In[76]:


with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# In[77]:


pipeline = joblib.load('model.pickle')


# In[78]:


new_obs_str = '{"age": 19, "workclass":0 , "education": 1, "marital-status": 0, "occupation": 1, "race": 1, "sex": 0, "capital-gain": 0,"capital-loss":0,"hours-per-week":25,"native-country":1}'
new_obs_dict = json.loads(new_obs_str)
obs = pd.DataFrame([new_obs_dict], columns=columns)
obs = obs.astype(dtypes)


# In[79]:


obs


# In[80]:


outcome = pipeline.predict_proba(obs)
outcome

