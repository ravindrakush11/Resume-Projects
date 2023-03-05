#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
'python-machine-learning-book-2nd-edition'
'/master/code/ch10/housing.data.txt', header = None, sep = '\s+')


# In[2]:


print(df)


# In[3]:


df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


# Exploratory data analysis

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols])      #height = 2.5
plt.tight_layout()
plt.show()


# Correlation

# In[5]:


import numpy as np

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, annot = True, xticklabels = cols, yticklabels = cols)
plt.show()


# In[27]:


from sklearn.linear_model import LinearRegression

X = df[['RM']].values  #.values
y = df[['MEDV']].values

slr = LinearRegression()
slr.fit(X, y)

print('Slope : %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


# slr = simple linear regression

# In[13]:


def lin_regplot(x, y, model):
    plt.scatter(x, y)
    plt.plot(x, model.predict(x))
    return (None)

lin_regplot(X, y, slr)
plt.xlabel('Average Number of Rooms', )
plt.ylabel('Price($)')
plt.show()


# Evaluating the Performance of linear regression models

# In[15]:


from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values  
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# In[16]:


from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))


# In[17]:


from sklearn.metrics import r2_score
print('R^2 train: %.3f  \nTest: %.3f' %(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


# In[28]:


from sklearn.linear_model import RANSACRegressor as RR

X = df[['RM']].values
y = df[['MEDV']].values

ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         loss='absolute_loss', 
                         residual_threshold=5.0,
                         random_state=0)
ransac.fit(X, y)


# In[29]:


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


# In[32]:



line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])   

plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o', label='Inliers')

plt.scatter(X[outlier_mask], y[outlier_mask],c='limegreen', edgecolor='white', marker='s', label='Outliers')

plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')

plt.show()


# In[31]:


print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)


# Ridge Regularize

# In[33]:


from sklearn.linear_model import Ridge

X = df[['RM']].values
y = df[['MEDV']].values

ridge = Ridge(alpha = 1.0)

ridge.fit(X, y)

def lin_regplot(x, y, model):
    plt.scatter(x, y)
    plt.plot(x, model.predict(x))
    return (None)

lin_regplot(X, y, slr)
plt.xlabel('Average Number of Rooms', )
plt.ylabel('Price($)')
plt.show()


# In[37]:


from sklearn.linear_model import Lasso

X = df[['RM']].values
y = df[['MEDV']].values

lasso = Lasso(alpha = 1.0)

lasso.fit(X, y)

def lin_regplot(x, y, model):
    plt.scatter(x, y)
    plt.plot(x, model.predict(x))
    return (None)

lin_regplot(X, y, lasso)
plt.xlabel('Average Number of Rooms', )
plt.ylabel('Price($)')
plt.show()


# In[42]:


from sklearn.linear_model import ElasticNet

X = df[['RM']].values
y = df[['MEDV']].values

elanet = ElasticNet(alpha = 1.0, l1_ratio=0.5)

elanet.fit(X, y)

def elanet_regplot(x, y, model):
    plt.scatter(x, y)
    plt.plot(x, model.predict(x))
    return (None)

lin_regplot(X, y, elanet)
plt.xlabel('Average Number of Rooms', )
plt.ylabel('Price($)')
plt.show()


# In[44]:


from sklearn.preprocessing import PolynomialFeatures
X = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

# create quadratic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# fit features
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))


# plot results
plt.scatter(X, y, label='training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='linear (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2, 
         linestyle=':')

plt.plot(X_fit, y_quad_fit, 
         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red', 
         lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit, 
         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green', 
         lw=2, 
         linestyle='--')

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')

#plt.savefig('images/10_11.png', dpi=300)
plt.show()


# In[45]:



# ## Random forest regression



X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)





forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[46]:


plt.scatter(y_train_pred,  
            y_train_pred - y_train, 
            c='steelblue',
            edgecolor='white',
            marker='o', 
            s=35,
            alpha=0.9,
            label='training data')
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='limegreen',
            edgecolor='white',
            marker='s', 
            s=35,
            alpha=0.9,
            label='test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('images/10_14.png', dpi=300)
plt.show()

