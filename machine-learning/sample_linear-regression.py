"""
regression_examples.py

Comprehensive implementations and usage examples for:
1. Multiple Linear Regression (from scratch & scikit-learn)
2. Logistic Regression (from scratch & scikit-learn)

Code style: PEP8, type hints, docstrings, clear structure.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression as SklearnLR, LogisticRegression as SklearnLogReg
from sklearn.datasets import fetch_california_housing, load_breast_cancer


class MultipleLinearRegression:
    """
    Multiple Linear Regression using Normal Equation & Gradient Descent.
    """

    def __init__(self,
                 lr: float = 0.01,
                 n_iters: int = 1000,
                 fit_intercept: bool = True):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def fit_normal(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit model using the Normal Equation: (X^T X)^{-1} X^T y
        """
        X_b = self._add_intercept(X)
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.coef_ = theta
            self.intercept_ = 0.0

    def fit_gd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit model using Batch Gradient Descent.
        """
        X_b = self._add_intercept(X)
        m, n = X_b.shape
        theta = np.zeros(n)

        for _ in range(self.n_iters):
            gradients = (1 / m) * X_b.T.dot(X_b.dot(theta) - y)
            theta -= self.lr * gradients

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.coef_ = theta
            self.intercept_ = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for X.
        """
        return X.dot(self.coef_) + self.intercept_


class LogisticRegression:
    """
    Binary Logistic Regression using Batch Gradient Descent.
    """

    def __init__(self,
                 lr: float = 0.01,
                 n_iters: int = 1000,
                 fit_intercept: bool = True):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_b = self._add_intercept(X)
        m, n = X_b.shape
        theta = np.zeros(n)

        for _ in range(self.n_iters):
            z = X_b.dot(theta)
            predictions = self._sigmoid(z)
            gradients = (1 / m) * X_b.T.dot(predictions - y)
            theta -= self.lr * gradients

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.coef_ = theta
            self.intercept_ = 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X.dot(self.coef_) + self.intercept_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


def main():
    # 多元线性回归示例（California housing）
    housing = fetch_california_housing()
    X_lin, y_lin = housing.data, housing.target
    X_train_lin, X_test_lin, y_train_lin, y_test_lin = \
        train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

    # 特征缩放
    scaler = StandardScaler()
    X_train_lin = scaler.fit_transform(X_train_lin)
    X_test_lin = scaler.transform(X_test_lin)

    # from scratch
    mlr = MultipleLinearRegression(lr=0.1, n_iters=1000)
    mlr.fit_gd(X_train_lin, y_train_lin)
    y_pred_lin = mlr.predict(X_test_lin)
    print("[Scratch] MLR MSE:", mean_squared_error(y_test_lin, y_pred_lin))
    print("[Scratch] MLR R2:", r2_score(y_test_lin, y_pred_lin))

    # scikit-learn
    skl_lr = SklearnLR()
    skl_lr.fit(X_train_lin, y_train_lin)
    y_pred_skl = skl_lr.predict(X_test_lin)
    print("[Sklearn] LR MSE:", mean_squared_error(y_test_lin, y_pred_skl))
    print("[Sklearn] LR R2:", r2_score(y_test_lin, y_pred_skl))

    # 逻辑回归示例
    cancer = load_breast_cancer()
    X_log, y_log = cancer.data, cancer.target
    X_train_log, X_test_log, y_train_log, y_test_log = \
        train_test_split(X_log, y_log, test_size=0.2, random_state=42)

    # 标准化
    X_train_log = scaler.fit_transform(X_train_log)
    X_test_log = scaler.transform(X_test_log)

    # from scratch
    logr = LogisticRegression(lr=0.1, n_iters=1000)
    logr.fit(X_train_log, y_train_log)
    y_pred_log = logr.predict(X_test_log)
    print("[Scratch] LogReg Accuracy:", accuracy_score(y_test_log, y_pred_log))
    print("[Scratch] LogReg Confusion Matrix:\n", confusion_matrix(y_test_log, y_pred_log))

    # scikit-learn
    skl_log = SklearnLogReg(max_iter=1000)
    skl_log.fit(X_train_log, y_train_log)
    y_pred_skl_log = skl_log.predict(X_test_log)
    print("[Sklearn] LogReg Accuracy:", accuracy_score(y_test_log, y_pred_skl_log))
    print("[Sklearn] LogReg Confusion Matrix:\n", confusion_matrix(y_test_log, y_pred_skl_log))


if __name__ == "__main__":
    main()
