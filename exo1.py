
import numpy as np
import matplotlib.pyplot as plt

def sample_dataset(n, p, q, r):
    """
        Sample a dataset of n samples according to
        the joint law.
        X ~ B(2, 1/4) Binomial Law
        Y ~ B(p) if X=0
        Y ~ B(q) if X=1
        Y ~ B(r) if X=2
    """
    X = np.random.randint(0, 3, n)
    Y = np.zeros(n)
    for i in range(n):
        if X[i] == 0:
            y_i = np.random.binomial(1, p)
        elif X[i] == 1:
            y_i = np.random.binomial(1, q)
        elif X[i] == 2:
            y_i = np.random.binomial(1, r)
        else:
            raise ValueError("incorrect value of X")
        Y[i] = y_i
    return X, Y

def compute_empirical_risk(f, X, Y):
    """
        Compute empirical risk of predictor
        on the dataset
        Parameters:
            X: 1D array
            Y: 1D array
            f: predictor
        Returns:
            empirical risk
        We use the "0-1"-loss
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples)
    for i in range(n_samples):
        predictions[i] = f(X[i])
    return (predictions!=Y).sum()/n_samples

def predictor(x):
    if x == 0:
        return 1
    elif x == 1:
        return 0
    elif x == 2:
        return 1
    else:
        raise ValueError("incorrect input")


p = 1/3
q = 2/3
r = 1/4
max_n_samples = 1500

generalization_error = (9/16)*(1-p) + (3/8)*(q) + (1/16)*(1 - r)
bayes_risk = (9/16)*min(p, 1-p) + (3/8)*min(q, 1-q) + (1/16)*min(r, 1-r)

empirical_risks = list()
for n in range(1, max_n_samples):
    X, Y = sample_dataset(n, p, q, r)
    empirical_risks.append(compute_empirical_risk(predictor, X, Y))
plt.plot(range(1, max_n_samples), empirical_risks, "o", markersize=2, alpha=0.3, label=r"$R_n(f)$"+" empirical risk")
plt.plot(range(1, max_n_samples), (max_n_samples-1)*[generalization_error], color="hotpink",label="real risk / generalization error")
plt.plot(range(1, max_n_samples), (max_n_samples-1)*[bayes_risk], color="red",label="bayes risk")
plt.xlabel("n")
plt.legend(loc="best")
plt.title(r"$f$"+": Empirical risk and generalization error and bayes risk"+"\n"+f"$R(f)$"+f"={generalization_error:.2f}")
plt.savefig("image/empirical_risk_and_generalization_error_and_bayes_risk.pdf")
plt.close()
