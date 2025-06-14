import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X + np.random.randn(100, 1) * 0.5
X_b= np.c_[np.ones((100,1)), X]
# Normal Equation
beta_noraml= ( np.linalg.inv(X_b.T@ X_b) )@X_b.T @y
beta = np.zeros((2,1))
omega = y- X_b @ beta
print("Normal Equation Coefficients:", beta_noraml.flatten())
# Gradient Descent:
def gradient_descent(X, y, beta, max_cycle=1000, alpha=0.01):
    cost_history = []
    for i in range(max_cycle):
        y_hat = X @ beta
        gradient = (1/len(y)) * X.T @ (y_hat - y)
        beta_new = beta -alpha *gradient
        if np.linalg.norm(beta_new - beta, ord=2) < 1e-6:
            break
        beta = beta_new
        cost = np.mean((X.dot(beta_new) - y) ** 2) / 2
        cost_history.append(cost)
    return beta,cost_history

beta_gd,cost_history = gradient_descent(X_b, y, beta, 5000, alpha=0.01)
print("Gradient Descent Coefficients:", beta_gd.flatten())
plt.plot(cost_history)
plt.xlabel("iteration times")
plt.ylabel("loss function")
plt.title("Gradient Descent Cost History")
plt.grid(True)
plt.show()
plt.plot(X, y, 'o', label='Data points')
plt.plot(X, X_b @ beta_noraml, 'r-', label='Normal Equation Fit')
plt.plot(X, X_b @ beta_gd, 'g--', label='Gradient Descent Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Comparison')
plt.legend()
plt.grid(True)
plt.show() 