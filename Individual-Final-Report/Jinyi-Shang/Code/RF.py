import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

dfboston=pd.read_csv('data3.csv')   #this data has been preprocessed.
pd.set_option('display.max_columns',None)

print(dfboston.shape)
print(dfboston.head())


y=dfboston.MEDV
del(dfboston['MEDV'])
X=dfboston

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(y_test.shape)

parameters = {'n_estimators':[100,150,200], 'max_depth':[3,5,10]}

RF=RandomForestRegressor(random_state=1)
model = GridSearchCV(RF, parameters)
model.fit(X_train,y_train)

#parameter select
print(model.best_params_)

#result
y_pred=model.predict(X_test)
print("MSE:",mean_squared_error(y_test, y_pred))
print('R square:',r2_score(y_test, y_pred))


#plot
plt.plot(range(127),y_pred,'b',marker='.',label='simulate')
plt.plot(range(127),y_test,'r',marker='.',label='true')
plt.title('true price vs simulate price')
plt.legend()
plt.show()


