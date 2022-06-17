"""
This file is the simulation of step 7 of the question 3, that gives an estimations of sigma2
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def generate_output_data(X, theta_star, sigma):
    """
        generate input and output data (supervised learning)
        according to the linear model, fixed design setup
        - X is fixed
        - Y is random, according to

        Y = Xtheta_star + epsilon

        where epsilon is a centered gaussian noise vector with variance
        sigma*In

        Parameters:
            X (float matrix): (n, d) design matrix
            theta_star (float vector): (d, 1) vector (optimal parameter)
            sigma (float): variance each epsilon

        Returns:
            Y (float matrix): output vector (n, 1)
    """
    n = X.shape[0]
    d = X.shape[1]
    noise = r.normal(0, sigma, size=(n, 1))
    Y = np.matmul(X, theta_star)+noise
    return Y

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

def estimator_sigma2(theta_hat, X, Y):
    """
        Compute the sigma2 estimation with parameter theta_hat,
        between Xtheta_hat and the labels Y

        Parameters:
            X (float matrix): (n, d) matrix
            Y (float vector): (n, 1) vector
            theta_hat (float vector): (d, 1) vector

        Returns:
            The estimation of sigma2 
    """
    n = X.shape[0]
    d = X.shape[1]
    Y_predictions = np.matmul(X, theta_hat)
    return 1/(n - d)*(np.linalg.norm(Y-Y_predictions))**2

r = np.random.RandomState(4)
n_list = np.arange(30, 2000, 1)
d = 10
theta_star = r.rand(d).reshape(d, 1)

sigma = 0.2
bayes_risk = sigma**2

mean_errors = []
for n in n_list:
        X = r.rand(n, d)
        Y = generate_output_data(X, theta_star, sigma)
        theta_hat = OLS_estimator(X, Y)
        mean_errors.append(estimator_sigma2(theta_hat, X, Y))
        
bayes_risk_list = [bayes_risk] * len(n_list)

plt.figure(figsize=(15, 8))
plt.plot(n_list, mean_errors, "o", label="$\sigma^2$ estimator")
plt.plot(n_list, bayes_risk_list, color="red", label="Bayes risk: $\sigma^2$")
plt.title("Estimation of $\sigma^2$")
plt.xlabel("n")
plt.ylabel("risk")
plt.legend()
plt.savefig("image/exo3/sigma2_estimation.pdf")