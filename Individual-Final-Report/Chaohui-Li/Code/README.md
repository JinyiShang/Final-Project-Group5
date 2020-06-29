# Contents
- Data Preprocessing
  - Load Data
  - Fill the Missing Values
  - Non-linear Regression
  - Standardization
  - Significance Test
- Linear Regression

# Data Preprocessing
This part will show the process of data preprocessing.

## Load Data and Summarize Data
'''
addr = 'datasets_HousingData.csv'
boston_housing = pd.read_csv(addr)
boston_housing.info()
boston_housing.drop_duplicates()
pd.set_option('display.max_columns', None)
boston_housing.describe()
'''

## Fill the Missing Values
1. Get features that exist missing values.
2. Get the distribution of features that exist missing values.
3. Use "median" method to fill the skewed distributed features.
   Use "mean" method to fill the normal distributed features.
   Use "mode" method to fill the dummy variable.
4. Get the new dataset that has filled the missing values.

'''
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

price = df['MEDV']
features = df.drop('MEDV', axis=1)
'''

## Non-linear Transformation
First, we obtain the distribution of every feature. Due to most features are skewed distributed, we choose a non-linear transformation to deal with data. We choose "Yeo-Johnson" transfer function.

## Standardization
We use the z-score normalization, which is also called standardization. After standardization, the mean value of the data is 0 and the standard deviation is 1.

'''
cols = list(df.columns)
for col in cols:
    sns.distplot(df[col])
    plt.legend([col])
    plt.show()
    
pt = preprocessing.PowerTransformer()
features_pt = pt.fit_transform(features)
'''

## Significance Test
We use the SelectKBset package to get the f-test value, mutual-info value and the responding p-value.

'''
f_test, _ = f_regression(features_pt, price)
mi = mutual_info_regression(features_pt, price)
new = SelectKBest(f_regression, k='all')
new.fit_transform(features_pt, price)
print('p_values of features are:', new.pvalues_)
plt.figure(figsize=(15, 5))
for i in range(len(cols)):
    plt.scatter(features_pt[:, i], price, edgecolors='black', s=20)
    plt.xlabel("{}".format(cols[i]), fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}, p_value={:.2f}".format(f_test[i], mi[i], new.pvalues_[i]))
    plt.show()
'''

# Linear Regression
We separate the dataset to train dataset and test dataset. We use the sklearn.Linear Regression method to achieve the least square errors.
Then we get the coefficients of linear models, mean square errors and R-square. Also, we drow the plot of comparision between true values and predicted values. 

'''
x_train, x_test, y_train, y_test = train_test_split(features_pt, price, random_state=1)

lr1 = LinearRegression()
lr1.fit(x_train, y_train)
y_pred1 = lr1.predict(x_test)
plt.plot(range(len(y_pred1)), y_pred1, 'r', label='y_predict')
plt.plot(range(len(y_test)), y_test, 'g', label='y_test')
plt.legend()
plt.xlabel("MSE:{}, R-square:{}".format(metrics.mean_squared_error(y_test, y_pred1), r2_score(y_test, y_pred1)), fontsize=14)
plt.title('sklearn: linear regression')
plt.show()
print('The coefficients of linear regression model are:', lr1.coef_)
print('The intercept of linear regression model is:', lr1.intercept_)
print("MSE:", metrics.mean_squared_error(y_test, y_pred1))
print("R^2:", r2_score(y_test, y_pred1))
'''
