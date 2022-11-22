#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
get_ipython().run_line_magic('pip', 'install --upgrade category_encoders')
from category_encoders import TargetEncoder
import joblib


# **Load player player_data set & print head**

# In[2]:


player_data=pd.read_csv("player-value-prediction.csv")
player_data.head()


# **Display some info about player_data set**

# In[3]:


player_data.describe()


# **check nulls in player_data**
# 

# In[4]:


print(player_data.isnull().sum())
print(player_data.isnull().values.sum())
f, ax = plt.subplots(figsize=(50, 6))
sns.heatmap(player_data.isnull(),yticklabels=False,cbar=False,cmap="viridis",ax=ax)


# In[5]:


data=player_data.copy()
data.head()


# In[6]:


#preprocessing of traits column
player_data['traits'] = player_data['traits'].fillna("Technical Dribbler (CPU AI Only)")# fill nulls with mod

traits_text=""
for i in player_data['traits']:
    traits_text=traits_text+','
    traits_text=traits_text+str(i)

trait_list = traits_text.split(',')

unique_list = []
for x in trait_list:
    if x not in unique_list:
        unique_list.append(x)

print(len(unique_list))


# **Nulls** **handling** 
# 
# 1.   wage & value ==> replace nulls with median(found that median =null ,so i replaced it with mode)
# 2.   national_rating,contract_end_year,club_join_date ==> fill nulls with zero
# 
# 1.   club_rating==> replace nulls with mean
# 
# 
# 
# 
# 4.   Drop columns['national_team','national_team_position','tags','club_team','club_position','traits','national_jersey_number','club_jersey_number'])]
# 2.   predict nulls of last 27 columns
# 
# 
# 
# 

# In[7]:


#Drop columns
data.drop(['national_team','national_team_position','tags',
                  'club_team','club_position','traits','national_jersey_number','club_jersey_number'],axis=1, inplace=True)
#fill null with zero
data.fillna({'national_rating':0,'club_join_date':0,'contract_end_year':0},inplace=True) 



#replace nulls with mode
data['value'].fillna(player_data.value.mode()[0],inplace=True)
data['wage'].fillna(player_data.wage.mode()[0],inplace=True)
data['release_clause_euro'].fillna(player_data.release_clause_euro.mode()[0],inplace=True)
#replace nulls with mean
data['club_rating'].fillna(int(data['club_rating'].mean()),inplace=True)

#predict nulls
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 10, weights = 'uniform', 
metric = 'euclidean', algorithm = 'kd_tree')
df=player_data.iloc[:,65:92].dropna()
X_train=np.array(df.iloc[:,-1])
X_train = X_train.reshape(X_train.shape[0],1)
df.drop(['value'],axis=1,inplace=True)


def impute_missing_occ (row):
    if pd.isnull(row[column_name]) :
        return knn_model.predict(
            row[['value']].values.reshape((-1,1)))
    else:
        return row[[column_name]]
  
for i,column_name in enumerate(df.columns):
    y_train=np.array(df.iloc[:,i])
    y_train = y_train.reshape(y_train.shape[0],1)
    knn_model.fit(X_train, y_train)
    data[column_name]=data.apply(impute_missing_occ,axis=1) 

print("Test save knn")
print(data[column_name])

f, ax = plt.subplots(figsize=(70, 6))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap="viridis",ax=ax)


# In[8]:


get_ipython().run_line_magic('pip', 'install researchpy')
import researchpy as rp

catCols = player_data.select_dtypes("object").columns
V=np.zeros([df.shape[1],df.shape[1]])
for i,column_name in enumerate(df.columns):
    v=[]
    for j,name in enumerate(df.columns): 
        ctab,chi_statistic,expected=rp.crosstab(player_data[name],player_data[column_name],margins=False,test='chi-square',expected_freqs=True)
        v.append(chi_statistic.iloc[2,1])
    V[i]=v 

categories_corr=pd.DataFrame(V,index=df.columns,columns=df.columns)   
categories_corr 


# **Handling categories**
# 
# 1.   work_rate,body_type ==> label_encoding
# 2.  preffered_foot ==> one hot encoding
# 
# 1.   positions ==> split by (,),then apply labe encoding
# 
# 2.   last 27 columns ==> target encoding
# 
# 
# 
# 
# 

# In[9]:


# Splitting strings
data['club_join_date'] = data['club_join_date'].astype(str)
for i,cell in enumerate(data['club_join_date']):
      
        if cell=="0":
            data['club_join_date'][i]=int(0)
        else:
            data['club_join_date'][i]=int(cell.split('/')[2])    
data['club_join_date'] = data['club_join_date'].astype(int)
            


data['contract_end_year'] = data['contract_end_year'].astype(str)
for i,cell in enumerate(data['contract_end_year']):
    if cell=="0":
            data['contract_end_year'][i]=int(0)
    elif len(cell)>4:      
            data['contract_end_year'][i]=int("20"+cell.split('-')[2])
            
data['contract_end_year'] = data['contract_end_year'].astype(int)      


for i,cell in enumerate(data['contract_end_year']):
    if data['contract_end_year'][i]==0 and data['club_join_date'][i]>0:
         data['club_join_date'][i]=0
    elif data['contract_end_year'][i]>0 and data['club_join_date'][i]==0:
         data['contract_end_year'][i]=0

# subtract contract_end_year from club_join_date
years_player_club=data['contract_end_year']-data['club_join_date']
data.insert(20,'years_player_club',years_player_club)
# Drop contract_end_year & club_join_date
data.drop(['contract_end_year','club_join_date'], axis=1, inplace=True)


# In[10]:


split_positions = data['positions'].str.split(',', expand = True).rename(columns = {0:"first_positions",1:"second_positions",2:"third_positions",3:"fourth_positions",})
split_positions = split_positions.fillna("0")  


# labelencoder = LabelEncoder()

# split_positions['label_first_pos'] =  labelencoder.fit_transform(split_positions['first_positions'])
# split_positions['label_second_pos'] =  labelencoder.fit_transform(split_positions['second_positions'])
# split_positions['label_third_pos'] =  labelencoder.fit_transform(split_positions['third_positions'])
# split_positions['label_fourth_pos'] =  labelencoder.fit_transform(split_positions['fourth_positions'])

# split_positions.drop('first_positions', axis=1, inplace=True)
# split_positions.drop('second_positions', axis=1, inplace=True)
# split_positions.drop('third_positions', axis=1, inplace=True)
# split_positions.drop('fourth_positions', axis=1, inplace=True)

# for i,c in enumerate(split_positions.columns):
#     data.insert(7+i,c,split_positions[c])

data.drop(['positions'],axis=1,inplace=True)


# In[11]:


def body_type_encoding(row):
    if row[["body_type"]].values == "Lean" :
        return 1;
    elif row[["body_type"]].values == "Normal" :
        return 2;  
    elif row[["body_type"]].values == "Stocky" :
        return 3;     
    else:
        return 2;

data["body_type"]=data.apply(body_type_encoding,axis=1)  


#work rate encoding
def work_rate_encoding(row):
    if row[["work_rate"]].values == "Low/ Low" :
        return 1;
    elif row[["work_rate"]].values == "Low/ Medium" :
        return 2;  
    elif row[["work_rate"]].values == "Medium/ Medium" :
        return 3;  
    elif row[["work_rate"]].values == "Low/ High" :
        return 4;
    elif row[["work_rate"]].values == "Medium/ Low" :
        return 5;
    elif row[["work_rate"]].values == "Medium/ High" :
        return 6;
    elif row[["work_rate"]].values == "High/ Low" :
        return 7;
    elif row[["work_rate"]].values == "High/ Medium" :
        return 8;
    elif row[["work_rate"]].values == "High/ High" :
        return 9;                           
    else:
        return 3;

data["work_rate"]=data.apply(work_rate_encoding,axis=1)


#one hot encoding
data['preferred_foot'].unique()
one_hot_encoder = OneHotEncoder()
preferred_foot_array = one_hot_encoder.fit_transform(data[['preferred_foot']]).toarray()
preferred_foot_labels = np.array(one_hot_encoder.categories_).ravel()#to make it an array, and .ravel() to convert it from array of arrays to array of strings
preferred_foot = pd.DataFrame(preferred_foot_array, columns=preferred_foot_labels)
v=data['value']
data.drop(['value','preferred_foot'],axis=1,inplace=True)
data = pd.concat([data, preferred_foot], axis = 1)
data = pd.concat([data, v], axis = 1)

#target encoding
X=player_data.iloc[:,65:91]


for i,c in enumerate(X.columns):
    encoder=TargetEncoder()
    data[c]=encoder.fit_transform(data[c],data['value'].astype('int'))
    
    

data.info()


# **Extract important features**
# 

# In[12]:


pd.set_option('display.max_columns', 500)
player_data.corr()


# In[13]:


feature_matrix=player_data.drop(['value'],axis=1)


# In[14]:


#with the following function we can select highly correlated features
#it will remove the first feature that is correlated with anything other features
def correlation(corr_matrix,threshold):
    col_corr=set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>=threshold:
                colname=corr_matrix.columns[i] #getting the name of the column
                col_corr.add(colname)
    return col_corr   


# In[15]:


corr_matrix=feature_matrix.corr()
corr_features=correlation(corr_matrix,0.8)
print(len(set(corr_features)))
corr_features


# In[16]:


corr_categories_features=correlation(categories_corr,1.0)
print(len(set(corr_categories_features)))
corr_categories_features


# In[17]:


df=data.copy()

data.drop(['id','name','full_name','birth_date','height_cm','nationality'],axis=1,inplace=True)
data.drop(corr_features,axis=1,inplace=True)
data.drop(corr_categories_features,axis=1,inplace=True)
print(data.shape)


# In[25]:


data.info()


# In[29]:


X=data.iloc[:,0:len(data.iloc[0,:])-1]#features
Y=data.iloc[:,-1]#label


# In[30]:


#feature scaling
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
joblib.dump(scaler,'scaler')
X.head()


# In[20]:


#split data int train 80%  test 20%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

print(X_train.shape)
print(X_test.shape)


# **Apply   polynomial/Normal/Regularized   regression model**

# In[21]:


import time
def model_trial(X_train, X_test, y_train, y_test, model, degree=30):
    start=time.time()
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)

    model.fit(X_train_poly, y_train)

    y_train_predicted = model.predict(X_train_poly)
    prediction = model.predict(poly_features.fit_transform(X_test))

    train_err = metrics.mean_squared_error(y_train, y_train_predicted)
    test_err = metrics.mean_squared_error(y_test, prediction)
    end=time.time()
    print('Train subset (RMSE) for degree {}: '.format(degree), np.sqrt(train_err))
    print('test subset (RMSE) for degree {}: '.format(degree), np.sqrt(test_err))
    print('Train data Accuracy',model.score(X_train_poly,y_train))
    print('Test data Accuracy',model.score(poly_features.fit_transform(X_test),y_test))
    print(f"Training time:{end - start}s")


# In[23]:


import time
import joblib

def model_trial_R(X_train, X_test, y_train, y_test, model, degree=30):
    start=time.time()
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)

    model.fit(X_train_poly, y_train)

    y_train_predicted = model.predict(X_train_poly)




    prediction = model.predict(poly_features.fit_transform(X_test))

    train_err = metrics.mean_squared_error(y_train, y_train_predicted)
    test_err = metrics.mean_squared_error(y_test, prediction)
    end=time.time()
    print('Train subset (RMSE) for degree {}: '.format(degree), np.sqrt(train_err))
    print('test subset (RMSE) for degree {}: '.format(degree), np.sqrt(test_err))
    print('Train data Accuracy',model.score(X_train_poly,y_train))
    print('Test data Accuracy',model.score(poly_features.fit_transform(X_test),y_test))
    print(f"Training time:{end - start}s")


# In[ ]:


def cross_validation(X_train, y_train, model,Cfold=10, degree=30):
    poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)
    # fit the transformed features to Linear Regression
    scores = cross_val_score(model, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=Cfold)
    model_2_score = abs(scores.mean())
    model.fit(X_train_poly, y_train)

    print("cross validation score is ",np.sqrt(model_2_score))


# In[ ]:


#tuning degree of polynomial regression
print("polynomial_model of degree 1 with cross validation")
cross_validation(X_train, y_train ,linear_model.LinearRegression(),10,1)
print("polynomial_model of degree 2 with cross validation")
cross_validation(X_train, y_train ,linear_model.LinearRegression(),10,2)


# In[ ]:


# best model with degree 2
print("polynomial model")
model_trial(X_train, X_test, y_train, y_test ,linear_model.LinearRegression(),2)


# In[ ]:


#tune degree with redge model
print('Ridge model')
cross_validation(X_train, y_train, linear_model.Ridge(),10,1)
cross_validation(X_train, y_train, linear_model.Ridge(),10,2)
cross_validation(X_train, y_train, linear_model.Ridge(),5,3)


# In[ ]:


#tuning alpha for ridge regression model using GridSearchCV

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train_poly,y_train)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[ ]:


# best model with degree 2 and alpha=0.01
print("Ridge model")
model_trial_R(X_train, X_test, y_train, y_test ,linear_model.Ridge(alpha=0.01),2)

