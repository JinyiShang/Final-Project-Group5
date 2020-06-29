import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import data_preprocessing

features_pt = data_preprocessing.features_pt
price = data_preprocessing.price

# train_set and test_set
x_train, x_test, y_train, y_test = train_test_split(features_pt, price, random_state=1)


# linear regression
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