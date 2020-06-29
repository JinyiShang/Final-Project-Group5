import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

def train_model(model):
    model.fit(X_train, y_train)

    # prediction
    pred = model.predict(X_test)

    # Show the results
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return mse, r2, model.loss_curve_, pred

boston = 'data3.csv'

# read data
boston = pd.read_csv(boston)

features = boston.iloc[:, 0:-1]
features = np.array(features)
target = boston.iloc[:, -1]
target = np.array(target)
rs = 1

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=rs)


# Best parameter selected by RandomizedSearchCV
hls = (100,)

paras = {'solver': 'sgd', 'max_iter': 11000, 'early_stopping': True, 'batch_size': 70, 'activation': 'relu'} # Parameters that selected by Randomized Search CV



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
plt.savefig('Error vs. Epochs.png')
plt.show()

# Real vs Fitted
plt.plot(range(len(y_test)), y_test, c='b')
plt.plot(range(len(prediction)), prediction, c='r')
plt.title('Real vs. Fitted')
plt.ylabel('Media Value of Home')
plt.legend(['Real', 'Fitted'])
plt.savefig('Real vs. Fitted.png')
plt.show()


'''
Results:

MSE: 10.35151226683004
R2: 0.8955009915342667
Number of iteration when the model converges: 150
'''







