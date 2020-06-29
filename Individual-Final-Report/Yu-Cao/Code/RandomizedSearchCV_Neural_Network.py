import numpy as np
import pandas as pd

import datetime

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings("ignore")


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


boston = 'data3.csv'

# read data
boston = pd.read_csv(boston)

features = boston.iloc[:, 0:-1]
features = np.array(features)
target = boston.iloc[:, -1]
target = np.array(target)

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

best_n_iter, best_score, paras = selepara_RSCV(NN, param_random, features, target)

end_t = datetime.datetime.now()

print((end_t - start_t).min) # calculate running time

print('Best n_iter:', best_n_iter)
print('RandomizedSearchCV best score:', best_score)
print('RandomizedSearchCV best parameters:', paras)

'''
Results:

/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (12000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (12000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (12000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (11000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (11000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (11000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (11000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (11000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (11000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (12000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (12000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (12000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (8000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (14000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (14000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (14000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (13000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (13000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (13000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (7000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
/Users/apple/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (9000) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
-999999999 days, 0:00:00
Best n_iter: 350
RandomizedSearchCV best score: 0.6223928954227341
RandomizedSearchCV best parameters: {'solver': 'sgd', 'max_iter': 11000, 'early_stopping': True, 'batch_size': 70, 'activation': 'relu'}

Process finished with exit code 0

'''








