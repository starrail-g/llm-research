import numpy as np

class GLM:
    """
    Generalized Linear Model implemented via Iteratively Reweighted Least Squares (IRLS).
    Supports families: 'gaussian', 'binomial', 'poisson'.
    """
    def __init__(self, family='gaussian', link='identity', max_iter=100, tol=1e-6):
        self.family = family
        self.link = link
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def _link(self, mu):
        if self.link == 'identity':
            return mu
        elif self.link == 'log':
            return np.log(mu)
        elif self.link == 'logit':
            return np.log(mu / (1 - mu))
        else:
            raise ValueError(f"Unknown link: {self.link}")

    def _link_inv(self, eta):
        if self.link == 'identity':
            return eta
        elif self.link == 'log':
            return np.exp(eta)
        elif self.link == 'logit':
            return 1 / (1 + np.exp(-eta))
        else:
            raise ValueError(f"Unknown link: {self.link}")

    def _variance(self, mu):
        if self.family == 'gaussian':
            return np.ones_like(mu)
        elif self.family == 'poisson':
            return mu
        elif self.family == 'binomial':
            return mu * (1 - mu)
        else:
            raise ValueError(f"Unknown family: {self.family}")

    def fit(self, X, y):
        # Add intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_samples, n_features = X.shape
        # Initialize coefficients
        beta = np.zeros(n_features)

        for iteration in range(self.max_iter):
            eta = X @ beta
            mu = self._link_inv(eta)
            var_mu = self._variance(mu)
            # Derivative of link: d_mu / d_eta
            if self.link == 'identity':
                mu_prime = np.ones_like(mu)
            elif self.link == 'log':
                mu_prime = mu
            elif self.link == 'logit':
                mu_prime = mu * (1 - mu)

            # Working weights and dependent variable
            z = eta + (y - mu) / mu_prime
            W = (mu_prime**2) / var_mu
            # Weighted least squares
            WX = X * W[:, np.newaxis]
            beta_new = np.linalg.pinv(WX.T @ X) @ (WX.T @ z)

            # Check convergence
            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                break
            beta = beta_new

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        eta = X @ np.concatenate([[self.intercept_], self.coef_])
        return self._link_inv(eta)


class LinearRegression:
    """
    Multivariate Linear Regression using Normal Equation or Gradient Descent.
    """
    def __init__(self, method='normal', lr=0.01, max_iter=1000, tol=1e-6):
        self.method = method
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Add intercept column
        X_b = np.hstack([np.ones((n_samples, 1)), X])

        if self.method == 'normal':
            # Closed-form solution: beta = (X^T X)^(-1) X^T y
            beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        elif self.method == 'gd':
            # Gradient Descent
            beta = np.zeros(n_features + 1)
            for i in range(self.max_iter):
                y_pred = X_b @ beta
                grad = (2 / n_samples) * (X_b.T @ (y_pred - y))
                beta_new = beta - self.lr * grad
                if np.linalg.norm(beta_new - beta, ord=2) < self.tol:
                    beta = beta_new
                    break
                beta = beta_new
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


# Example usage:
# X = np.random.randn(100, 3)
# beta_true = np.array([1.5, -2.0, 0.5])
# y = X @ beta_true + np.random.randn(100) * 0.1
# lr = LinearRegression(method='normal')
# lr.fit(X, y)
# print(lr.intercept_, lr.coef_)

# glm = GLM(family='poisson', link='log')
# y_poisson = np.random.poisson(lam=np.exp(X @ beta_true))
# glm.fit(X, y_poisson)
# print(glm.intercept_, glm.coef_)
