import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import r2_score
import datetime
import warnings
warnings.filterwarnings("ignore")


# Data Preprocess
# load data
addr = 'datasets_HousingData.csv'
boston_housing = pd.read_csv(addr)
boston_housing.info()
boston_housing.drop_duplicates()
pd.set_option('display.max_columns', None)
boston_housing.describe()

# missing values
result_isnull = boston_housing.isnull().any()
num_isnull = boston_housing.isnull().sum()
result_columns = boston_housing.columns[result_isnull]

for column in result_columns:
    sns.distplot(boston_housing[column])
    plt.legend([column])
    plt.show()


imp1 = SimpleImputer(missing_values=np.nan, strategy='median')
imp2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp3 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df = boston_housing.copy()
imp1.fit(np.array(df['CRIM']).reshape(-1, 1))
df['CRIM'] = imp1.transform(np.array(boston_housing['CRIM']).reshape(-1, 1))
imp1.fit(np.array(df['ZN']).reshape(-1, 1))
df['ZN'] = imp1.transform(np.array(boston_housing['ZN']).reshape(-1, 1))
imp2.fit(np.array(df['INDUS']).reshape(-1, 1))
df['INDUS'] = imp2.transform(np.array(boston_housing['INDUS']).reshape(-1, 1))
imp3.fit(np.array(df['CHAS']).reshape(-1, 1))
df['CHAS'] = imp3.transform(np.array(boston_housing['CHAS']).reshape(-1, 1))
imp1.fit(np.array(df['AGE']).reshape(-1, 1))
df['AGE'] = imp2.transform(np.array(boston_housing['AGE']).reshape(-1, 1))
imp1.fit(np.array(df['LSTAT']).reshape(-1, 1))
df['LSTAT'] = imp1.transform(np.array(boston_housing['LSTAT']).reshape(-1, 1))

# get data after filling the missing values
price = df['MEDV']
features = df.drop('MEDV', axis=1)

# plot of relationship between target and features
cols = list(df.columns)
for col in cols:
    sns.distplot(df[col])
    plt.legend([col])
    plt.show()


# power_transformer-best
pt = preprocessing.PowerTransformer()
features_pt = pt.fit_transform(features)

# the distribution of transformed data
cols.pop()

# feature selection
# significant test
f_test, _ = f_regression(features_pt, price)
mi = mutual_info_regression(features_pt, price)
new = SelectKBest(f_regression, k='all')
new.fit_transform(features_pt, price)
print('p_values of features are:', new.pvalues_)
for i in range(len(cols)):
    plt.scatter(features_pt[:, i], price, edgecolors='black', s=20)
    plt.xlabel("{}".format(cols[i]), fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}, p_value={:.2f}".format(f_test[i], mi[i], new.pvalues_[i]))
    plt.show()


# train_set and test_set
x_train, x_test, y_train, y_test = train_test_split(features_pt, price, random_state=1)

# linear regression
print('\nLinear regression:')
lr1 = LinearRegression()
lr1.fit(x_train, y_train)
y_pred1 = lr1.predict(x_test)
plt.plot(range(len(y_pred1)), y_pred1, 'r', label='y_predict')
plt.plot(range(len(y_test)), y_test, 'g', label='y_test')
plt.legend()
plt.xlabel("MSE:{}, R-square:{}".format(metrics.mean_squared_error(y_test, y_pred1), r2_score(y_test, y_pred1)),
           fontsize=14)
plt.title('sklearn: linear regression')
plt.show()


print('The coefficients of linear regression model are:', lr1.coef_)
print('The intercept of linear regression model is:', lr1.intercept_)
print("MSE:", metrics.mean_squared_error(y_test, y_pred1))
print("R^2:", r2_score(y_test, y_pred1))

# Neuron Network
'''
# parameter select
def selepara_RSCV(Model, params, X, Y):
    n_iter = np.arange(200, 400, 50)
    score = 0
    for i in n_iter:
        clf = RandomizedSearchCV(Model, params, n_iter=i, n_jobs=-1, cv=5)
        grid_result = clf.fit(X, Y)
        if grid_result.best_score_ > score:
            score = grid_result.best_score_
            parameters = grid_result.best_params_
            n = i
    return n, score, parameters


NN = MLPRegressor()
# Parameters
hls = (100,)
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_random = dict(
    activation=['identity', 'logistic', 'tanh', 'relu'],
    solver=['lbfgs', 'sgd', 'adam'],
    max_iter=np.arange(7000, 15000, 1000),
    batch_size=np.arange(10, 200, 10),
    early_stopping=[False, True])
start_t = datetime.datetime.now()
best_n_iter, best_score, paras = selepara_RSCV(NN, param_random, features_pt, price)
end_t = datetime.datetime.now()
# calculate running time
print((end_t - start_t).min)
# results
print('Best n_iter:', best_n_iter)
print('RandomizedSearchCV best score:', best_score)
print('RandomizedSearchCV best parameters:', paras)
'''

# neuron network
print('\nNueron Network:')


def train_model(model):
    model.fit(x_train, y_train)
    # prediction
    pred = model.predict(x_test)
    # Show the results
    mse = metrics.mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return mse, r2, model.loss_curve_, pred


hls = (100,)
paras = {'solver': 'sgd', 'max_iter': 11000, 'early_stopping': True, 'batch_size': 70,
         'activation': 'relu'}  # Parameters that selected by Randomized Search CV
# Train the model
nn = MLPRegressor(hidden_layer_sizes=hls,
                  activation=paras['activation'],
                  solver=paras['solver'],
                  max_iter=paras['max_iter'],
                  early_stopping=paras['early_stopping'],
                  batch_size=paras['batch_size'],
                  random_state=50)
MSE, R2, Loss_curve, prediction = train_model(nn)
print('MSE:', MSE)
print('R2:', R2)
print('Number of iteration when the model converges:', nn.n_iter_)
# Plot results
# Error and Epochs
pd.DataFrame(Loss_curve).plot()
plt.title('Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend('MSE')
plt.show()


# Real vs Fitted
plt.plot(range(len(y_test)), y_test, c='b')
plt.plot(range(len(prediction)), prediction, c='r')
plt.title('Real vs. Fitted')
plt.ylabel('Media Value of Home')
plt.legend(['Real', 'Fitted'])
plt.show()


# Random Forest
print('\nRandom forest:')
parameters = {'n_estimators': [100, 150, 200], 'max_depth': [3, 5, 10]}
RF = RandomForestRegressor(random_state=1)
model = GridSearchCV(RF, parameters)
model.fit(x_train, y_train)
# parameter select
print(model.best_params_)
# result
y_pred = model.predict(x_test)
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print('R square:', r2_score(y_test, y_pred))
# plot
plt.plot(range(127), y_pred, 'b', marker='.', label='simulate')
plt.plot(range(127), y_test, 'r', marker='.', label='true')
plt.title('true price vs simulate price')
plt.legend()
plt.show()


# Gradient Boosting
print('\nGradient Boosting:')
parameters = {'min_samples_leaf': [1, 3, 5, 10], 'n_estimators': [100, 150, 200, 250]}
reg = GradientBoostingRegressor(random_state=1)
model = GridSearchCV(reg, parameters)
model.fit(x_train, y_train)
# parameter select
print(model.best_params_)
# result
y_pred = model.predict(x_test)
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print('R square:', r2_score(y_test, y_pred))
# plot
plt.plot(range(127), y_pred, 'b', marker='.', label='simulate')
plt.plot(range(127), y_test, 'r', marker='.', label='true')
plt.title('true price vs simulate price')
plt.legend()
plt.show()
