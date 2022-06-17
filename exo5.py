"""
This file preforms several classification models on the dataset.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# import datas
X = np.load("data/classification/inputs.npy")
y = np.ravel(np.load("data/classification/labels.npy"))

# define knn
def knn_predict(X_train, y_train, X_test, k):
    """
        Predict with knn estimation

        Parameters:
            k (integer): number of nearest neighbors used.
            X_train (float matrix): samples in input space
            y_train (float vector): values of the target function
            
            X_test (float matrix): (n_samples, d) data for which we
            predict a value based on the dataset.

        Returns:
            y_predictions (float matrix): predictions for the data
            in x_test.

    """
    n_test = X_test.shape[0]
    y_predictions = np.zeros(n_test)
    dist = cdist(X_test, X_train, 'euclidean')
    for i in range(n_test):
        sort = np.argsort(dist[i])[:k]
        votes = np.sum(y_train[sort])
        if (votes <= 0):
            y_pred = -1
        else:
            y_pred = 1
        y_predictions[i] = y_pred
    
    return np.array(y_predictions)

# define error function: mean squared error 
def error(X_train, y_train, X_test, y_test, k):
    """
        Compute the errors
        
        Parameters:
            k (integer): number of nearest neighbors used.
            X_train (float matrix): samples in input space
            y_train (float vector): values of the target function
            
            X_test (float matrix): (n_samples, d) data for which we
            predict a value based on the dataset.
            
        Returns:
            mean squared error
    """
    y_predictions = knn_predict(X_train, y_train, X_test, k)
    d = y_test.shape[0]
    mean_squared_error = np.linalg.norm(y_predictions - y_test)**2 / d
    return mean_squared_error


mean_squared_errores = list()
max_errores = list()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
neighbors_list = np.arange(1, 100, dtype=int)
for k in neighbors_list:
    mean_squared_error= error(X_train, y_train, X_test, y_test, k)
    mean_squared_errores.append(mean_squared_error)
    
# Plot mean_squared_error against n_neighbors
plt.figure(figsize=[20, 10])
plt.plot(neighbors_list, mean_squared_errores, "o", label=f"k={k}")
plt.ylabel("mean squared error")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig("image/exo5/mean_squared_error_against_n_neighbors.png")

# user defined knn prediction
y_predictions = knn_predict(X_train, y_train, X_test, k=23)
print(f"The accuracy of user defined KNN is {accuracy_score(y_test, y_predictions)}")

# define sklearn knn
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 100)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(X_train, y_train)
print(f"The best n_neighbors found by GridSearchCV is: {knn_gscv.best_params_}")
y_pred_knn_cv = knn_gscv.predict(X_test)
print(f"The accuracy of sklearn KNN is {accuracy_score(y_test, y_pred_knn_cv)}")

# define LogisticRegression
LogisticRegressionClf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred_lrc = LogisticRegressionClf.predict(X_test)
print(f"The accuracy of Logistic Regression is {accuracy_score(y_test, y_pred_lrc)}")

# define svc
param_grid_svc = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
svc_gscv = GridSearchCV(SVC(), param_grid_svc, verbose=2)
svc_gscv.fit(X_train,y_train)
print(f"The best hyperparameters for SVC are: {svc_gscv.best_params_}")

y_pred_svc = svc_gscv.predict(X_test)
print(f"The accuracy of SVC is {accuracy_score(y_test, y_pred_svc)}")
