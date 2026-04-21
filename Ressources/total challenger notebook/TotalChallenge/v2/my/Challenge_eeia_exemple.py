#!/usr/bin/env python
# coding: utf-8

# # Challenge EEIA

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")


# # Load data

# Here we define the directory path where the data is stored and will be stored. The files train.csv and test.csv should be in that directory.

# In[2]:


data_path = ''


# In[3]:


data = pd.read_csv(f'{data_path}train.csv',delimiter=';',decimal=',',na_values=["#VALEUR!"],index_col="time")
data.index = pd.to_datetime(data.index, format='%d/%m/%Y %H:%M')

test = pd.read_csv(f'{data_path}test.csv',delimiter=';',decimal=',',na_values=["#VALEUR!"],index_col="time")
test.index = pd.to_datetime(test.index, format='%d/%m/%Y %H:%M')


# The following cell displays the first rows of the dataframe.

# In[4]:


data.head()


# In[5]:


test.head()

test.info()


# Getting the shape of the dataframe

# In[6]:


data.shape, test.shape


# The training set contains 368591 rows andd 12 columns and the test set contains 156960 rows and 12 columns. The test set contains almost half less data than the training set. All the values in the column Net Power (MW) are NaN and should be filled with our model's predictions.
# 
# Here are the list of the columns of both dataset : 

# <table border="0" padding="0" margin="0">
# <tbody>
# <tr><td>Amb temp (°C)</td><td>Température ambiante</td></tr>
# <tr><td>Comp inlet temp (°C)</td><td>Température entrée compresseur</td></tr>
# <tr><td>amb pressure</td><td>Pression ambiante</td></tr>
# <tr><td>HR %</td><td>%Humidité relative ambiante</td></tr>
# <tr><td>C/H</td><td>Rapport Carbonne/hydrogène du Gaz Nat</td></tr>
# <tr><td>Network Frequency (Hz)</td><td>Fréquence du réseau électrique en Hz</td></tr>
# <tr><td>Lower Heating Value (Wh/Nm3) &nbsp;</td><td>le pouvoir calorifique inférieur du Gaz Nat</td></tr>
# <tr><td>EOH (h)</td><td>Heures d'Opérations Equivalentes </td></tr>
# <tr><td>DP filtre</td><td>Perte de charge au niveau des filtres d'air entrée turbine à gaz</td></tr>
# <tr><td>CTRL anti givrage</td><td>Control de la vanne d'ouverture de l'anti-givre entrée turbine gaz</td></tr>
# <tr><td>IGV %</td><td>% d'ouverture de la valve IGV (Inlet Guide Vanes) afin de controller la charge de la CCGT</td></tr>
# <tr><td>Net Power (MW)</td><td>Production d'électricité NETTE générée par la CCGT</td></tr>
# </tbody>
# </table>

# # Checks

# We check that this data has the exact types we are waiting for.

# In[7]:


assert data.dtypes.equals(pd.Series(dict(zip(data.columns,["float64"]*len(data.columns)))))

assert test.dtypes.equals(pd.Series(dict(zip(test.columns,["float64"]*len(test.columns)))))


# Statistical metrics of the training set such as the minimum, average, standard deviation, maximum, and quantiles are computed below.

# In[8]:


data.head()


# In[9]:


data.describe()


# # EDA and first models

# In[10]:


data.isna().sum()


# There are 3388 NaN values in the dataframe.

# ### Imputing strategy

# In[11]:


from sklearn.impute import SimpleImputer


# In[12]:


# Dropping all the NaN values
# data.dropna(inplace=True)

# Replace the Nans using a given strategy

chosen_strategy = 'median' # "mean" / "constant" / "most_frequent"
for col in data.columns:
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    data[col] = imp_mean.fit_transform(data[[col]]).squeeze()


# In[13]:


assert (data.isna().sum(axis=0).sum() == 0)


# Let's start by checking all the variables

# In[14]:


fig, ax = plt.subplots(figsize=(20,16),nrows=3, ncols=4)

for i,col in enumerate(data.columns):
    data.head(300)[col].plot(ax=ax[i//4,i%4],title=col)


# In[15]:


data.hist(figsize=(20,16))


# ## Net Power (MW)

# In[16]:


data["Net Power (MW)"].plot(figsize=(20,12))


# Let's see if there is any correlation between the Net power (MW) and speed by taking a look at one of them, function of the other.

# In[17]:


x_var = "Amb temp °C"
y_var = "Net Power (MW)"
# data.plot.scatter(x=x_var,y=y_var)


# In[18]:


fig, ax = plt.subplots(figsize=(20,16),nrows=3, ncols=4)

for i,col in enumerate(data.columns):
    data.plot.scatter(x=col,y=y_var,ax=ax[i//4,i%4],title=col)


# The correlation between the variables is not obvious expect for the variables 'DP filtre' and 'IGV %'. Let's take a closer look at the values of the correlations.

# In[19]:


data.corr()


# In[20]:


data.corr().style.background_gradient()


# 'DP filtre' and 'IGV %' are highly correlated with the target variable.

# For a sample, the correlation is defined by : 
#     

# $$
# \frac{\sum \limits _{i=1} ^{n} (x_{i} - \bar x) (y_{i} - \bar y)}{\sqrt{\sum \limits _{i=1} ^{n}(x_{i} - \bar x)^{2}}\sqrt{\sum \limits _{i=1} ^{n}(y_{i} - \bar y)^{2}}} 
# $$

# What is important to recall is that it is comprised in the range $[-1, 1]$ and : <br>
#     - it is equal to 1 if the two variables are exactly the same <br>
#     - it is equal to -1 if the two variables are the exact opposite <br>
#     - when it is equal to 0, the two variables have nothing in common : they are independent one from the other, for example this could be the value of the bitcoin and the average wind speed in south korea, we know these two have nothing in common.<br>
#     - when it is > 0, the two variables are positively correlated, this means that on average, when one goes up, the other goes up too.<br>
#     - when it is < 0, the two variables are negatively correlated, this means that on average, when one goes up, the other goes down.<br>

# #### EDA ideas
# 
# - features selection: drop useless variables? why? how?
# - handle the nan values differently
# - create new variables ? 
# 
# 

# ## Modeling

# Based on this first EDA, a very simple model we can try to predict our sample is to try a linear model : 

# In order to do so, we import some libraries that will be useful.

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# We now cut our dataframe in two : one dataframe will be used for training, and the other one will be used to estimate what is the value of this first model we have made. 
# For this, why do we not directly use the test set ? The reason is that for the test set, we do not know the exact value of the power measurement.

# In[22]:


X = data.drop(columns="Net Power (MW)")
y = data["Net Power (MW)"]

print(X.shape, y.shape)


# In[23]:


X_train, X_eval, y_train, y_eval = train_test_split(X, y,test_size=0.2, shuffle=True,random_state=0)


# In[24]:


def evaluate_model(model):
    #print("Model name: ", type(model).__name__)
    #print("Model parameters: ", model.get_params())

    # Printing model accuracy
    model_train_mae = mean_absolute_error(y_train,model.predict(X_train))
    model_test_mae = mean_absolute_error(y_eval,model.predict(X_eval))

    #print("Model Mean Absolute error on the train set : %.2f" % model_train_mae)
    #print("Model Mean Absolute error on the test set : %.2f" % model_test_mae)

    return model_train_mae , model_test_mae


# In[25]:


model = LinearRegression()
model.fit(X_train, y_train)
evaluate_model(model)


# Nice ! We have our first model and it gives an error of 10.27 !..
# 
# Now wait, what is the value of that first model ? How can we know if 10.27 is actually a good error ? Well for this, a very neat way to be able to know if our model is worth anything is to compare it to a naive model. A naive model can be for example to predict everytime the same value, whatever the conditions. One of these naive model we have at hand would be to predict the mean value of the wind power in the train set. Let's see what would this model give. 

# Yes ! Good news, our model did really learn something good ! We are a lot better than the 'mean' or 'median' model, around 10 times better, based on this metric.

# # Next steps Modeling

# We have already seen first models above : the linear model with all the variables , and two naive models (median, and mean). It will be your job from now on to determine the best model, but let's already take a look at one classic model that data scientists usually try on for nearly any subject : Random Forest. 

# In[26]:


from sklearn.ensemble import RandomForestRegressor


# In[27]:


# Tracé

def figure(x, y):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue', label='Tracé')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Fonction de perte')
    plt.legend()


# In[28]:


max_samples=100000
x=[]
y=[]
yy=[]

for i in range(1,max_samples + 1,2000):
    rf_model = RandomForestRegressor(n_jobs=--1,max_samples=i)
    rf_model.fit(X_train, y_train)
    evaluate_model_train , evaluate_model_test = evaluate_model(rf_model)
    
    x.append(i)
    y.append(evaluate_model_test)
    yy.append(evaluate_model_train)

    print("itération : ", i, "evaluate_model_test : ", evaluate_model_test, "evaluate_model_train : ",evaluate_model_train)

figure(x, y)
figure(x, yy)


# Random Forest does better than the linear model.

# In[37]:


#from sklearn.model_selection import cross_val_score, KFold

print(x,"\n\n\n\n\n",y,"\n\n\n\n\n",yy)


# In[40]:


xx = x [4:]
yyy = y [4:]
yyyyyy = yy [4:]

print(xx,"\n\n\n\n\n",yyy,"\n\n\n\n\n",yyyyyy)

figure(xx,yyy)

figure (xx,yyyyyy)


# ## Cross Validation

# In[30]:


#cv_rf_model = RandomForestRegressor(n_jobs=-1,max_samples=200)


# In[31]:


"""
cv_scores = []
for i in range(2, 11):
    kfold = KFold(n_splits=i)
    scores = cross_val_score(cv_rf_model, X, y,scoring="neg_mean_absolute_error", cv=kfold)
    cv_scores.append(-scores.mean())

best_cv = cv_scores.index(max(cv_scores)) + 2
print(f"The best cross-validation split is {best_cv}-fold with a mean score of {max(cv_scores)}")
"""


# In[32]:


#X_train, X_eval, y_train, y_eval = train_test_split(X, y,test_size=1/best_cv, shuffle=True,random_state=0)


# ### Modeling ideas
# 
# - Try different parameters for the models ?
# - Try new models ?
# - Tune parameters ?
# 

# # Predictions on test set

# Now our model is fit, we can pass on to the predictions.
# 
# _Note: be careful when generating your submission file. Indeed, it needs to be a csv file with ";" as separator._

# In[33]:


#selected_model = rf_model #cv_rf_model


# In[34]:


"""X_test = test.drop(columns="Net Power (MW)")

df_predictions = pd.DataFrame({
    'time': test.index,
    'Net Power (MW)': selected_model.predict(X_test),
})

df_predictions.to_csv('predictions.csv', date_format='%d/%m/%Y %H:%M',index=False, sep=';')
df_predictions.head()"""


# Now it is your turn, what better model can you think of ?
