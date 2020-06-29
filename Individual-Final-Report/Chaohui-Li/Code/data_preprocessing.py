import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


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
plt.figure(figsize=(15, 5))
for i in range(len(cols)):
    plt.scatter(features_pt[:, i], price, edgecolors='black', s=20)
    plt.xlabel("{}".format(cols[i]), fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}, p_value={:.2f}".format(f_test[i], mi[i], new.pvalues_[i]))
    plt.show()
