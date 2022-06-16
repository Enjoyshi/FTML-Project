import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# import datas
X = np.load("data/regression/inputs.npy")
y = np.load("data/regression/labels.npy")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Define lists of hyperparameters
llambda_list = [10**(n) for n in np.arange(-8, 5, 0.2)]
alpha_list = [10**(n) for n in np.arange(-8, 5, 0.2)]

# Ridge Cross Validation for plot 
ridge_scores = []
ridge = Ridge()
for llambda in llambda_list:
    ridge.alpha = llambda
    val = np.mean(cross_val_score(ridge, X, y, cv = 10))
    ridge_scores.append(val)

# Lasso Cross Validation for plot 
lasso_scores = []
lasso = Lasso(tol=0.1)
for alpha in alpha_list:
    lasso.alpha = alpha
    val = np.mean(cross_val_score(lasso, X, y, cv = 10))
    lasso_scores.append(val)
    
# Plot cross_val_score against hyperparametres
plt.figure(figsize=(8, 8))
plt.xscale('log')
plt.plot(llambda_list, ridge_scores, marker = '.', label = "Ridge")
plt.plot(alpha_list, lasso_scores, marker = 'x', label = "Lasso")
plt.xlabel('hyperparameter')
plt.ylabel('cross_val_score')
plt.legend()
plt.savefig("image/exo4/cross_val_score_against_hyperparametres.pdf")
#plt.show()

# Ridge Hypertuning
params = {'alpha': llambda_list} 
ridge = Ridge()
ridge_model = GridSearchCV(ridge, params, cv = 10)
ridge_model.fit(X_train, y_train)
print(f"The best hyperparameter for ridge regression is: {ridge_model.best_params_}")

y_pred_ridge = ridge_model.predict(X_test)
print(f"The R2 score of ridge regression is {r2_score(y_test, y_pred_ridge)}")

# Lasso Hypertuning
params = {'alpha': alpha_list}
lasso = Lasso(tol=0.1)
lasso_model = GridSearchCV(lasso, params, cv = 10)
lasso_model.fit(X_train, y_train)
print(f"The best hyperparameter for lasso regression is: {lasso_model.best_params_}")

y_pred_lasso = lasso_model.predict(X_test)
print(f"The R2 score of lasso regression is {r2_score(y_test, y_pred_lasso)}")

def OLS_estimator(X, Y):
    """
        Compute the OLS estimator from the data.

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector

        Returns:
            OLS (float vector): (d, 1) vector
    """
    covariance_matrix = np.matmul(np.transpose(X), X)
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = np.matmul(inverse_covariance, np.matmul(np.transpose(X), Y))
    return theta_hat

print(f"The R2 score of OLS is {r2_score(y_test, np.matmul(X_test,OLS_estimator(X_train, y_train)))}")
