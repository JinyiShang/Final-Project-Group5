# Neural Network

## Content
* [Load Data and Split Train and Test Data](#Load-Data-and-Split-Train-and-Test-Data)  
* [RandomizedSearchCV](#RandomizedSearchCV)  
* [Train the Model to get MSE and R2](#Train-the-Model-to-get-MSE-and-R2)  
* [Plot results](#Plot-results)  
  * [Error and Epochs](#Error-and-Epochs)  
  * [Real Value and Fitted Value](#Real-Value-and-Fitted-Value) 

## Load Data and Split Train and Test Data
Firstly, read the preprocessed data, taking the last column 'MEDV' as target, and the rest columns as features.
Then, split data into train and test data.

```
boston = 'data3.csv'

boston = pd.read_csv(boston)

features = boston.iloc[:, 0:-1]
features = np.array(features)
target = boston.iloc[:, -1]
target = np.array(target)
rs = 1

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=rs)
```

## RandomizedSearchCV
RandomizedSearchCV is a grid search for parameters by sampling in the parameter space. It runs faster than GridSearchCV.
And its search ability depends on the set 'n_iter' parameter.  
So, I define a function named 'selepara_RSCV()' to run RandomizedSearchCV() with different n_iter.
And it returns n_iter, best_score and best_params.  
  
*Note*: This part is writen in a single file named 'RandomizedSearchCV_Neural_Network.py' due to time consuming of running it.
```
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
```  
The best parameters are:
**Parameters** | **activation** | **solver** | **batch_size** |  **max_iter** |  **early_stopping**  
---- | ---- | ---- | ---- | ---- | ----  
**Value** | relu | sgd | 70 | 11000 | True  

## Train the Model to get MSE and R2
Train the MLPRegressor() model with parameters got above.
```
nn = MLPRegressor(hidden_layer_sizes=hls,
                  activation=paras['activation'],
                  solver=paras['solver'],
                  max_iter=paras['max_iter'],
                  early_stopping=paras['early_stopping'],
                  batch_size=paras['batch_size'],
                  random_state=50)
```
Use the defined function named 'train_model()' to get MSE, R2, prediction.
```
def train_model(model):
    model.fit(x_train, y_train)
    # prediction
    pred = model.predict(x_test)
    # Show the results
    mse = metrics.mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return mse, r2, model.loss_curve_, pred
```
```
MSE, R2, Loss_curve, prediction = train_model(nn)
```

## Plot results

### Error and Epochs
Plot the MSE and Epochs image to see whether and when the model becomes convergence.
```
pd.DataFrame(Loss_curve).plot()
plt.title('Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend('MSE')
plt.savefig('Error vs. Epochs.png')
plt.show()
```

### Real Value and Fitted Value
Plot the real house price with the fitted house price to see how the model perform.
```
plt.plot(range(len(y_test)), y_test, c='b')
plt.plot(range(len(prediction)), prediction, c='r')
plt.title('Real vs. Fitted')
plt.ylabel('Media Value of Home')
plt.legend(['Real', 'Fitted'])
plt.savefig('Real vs. Fitted.png')
plt.show()
```
