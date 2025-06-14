# 线性回归学习笔记

> **范围**：从一元线性回归到最广义的多元线性回归（不含源码）

---

## 1. 基础概念

- **回归**：利用一个或多个自变量（特征）预测一个因变量（目标）。
    
- **一元线性回归**：单个特征 $x$ 与目标 $y$ 之间的线性关系。  
     $y = \theta_0 + \theta_1 x + \varepsilon$
    
- **多元线性回归**：多个特征 $\mathbf{x}=[x_1, x_2, \dots, x_n]$ 与目标 $y$ 之间的线性关系。  
    $y = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n + \varepsilon$
    

---

## 2. 模型表示

### 2.1. 矩阵形式

- 样本数：$m$，特征维度：$n$
    
- **设计矩阵**：  
    $$X = \begin{pmatrix}
    1 & x_{1,1} & x_{1,2} & \dots & x_{1,n} \
    1 & x_{2,1} & x_{2,2} & \dots & x_{2,n} \
    \vdots & \vdots & \vdots & & \vdots \
    1 & x_{m,1} & x_{m,2} & \dots & x_{m,n}
    \end{pmatrix} \in \mathbb{R}^{m\times(n+1)}$$
    
- **参数向量**：  
    $\boldsymbol{\theta} = [\theta_0, \theta_1, \dots, \theta_n]^\top \in \mathbb{R}^{(n+1)\times1}$
    
- **目标向量**：  
    $\mathbf{y} = [y_1, y_2, \dots, y_m]^\top \in \mathbb{R}^{m\times1}$
    
- **预测**：  
    $\hat{\mathbf{y}} = X,\boldsymbol{\theta} \in \mathbb{R}^{m\times1}$
    

---

## 3. 损失函数（MSE）

- 均方误差：  
    $J(\boldsymbol{\theta}) = \frac{1}{2m} \lVert X\boldsymbol{\theta} - \mathbf{y} \rVert^2 = \frac{1}{2m}(X\boldsymbol{\theta}-\mathbf{y})^\top (X\boldsymbol{\theta}-\mathbf{y})$
    
- **展开成三项**：  
    $J = \frac{1}{2m}\bigl(\boldsymbol{\theta}^\top X^\top X\boldsymbol{\theta} - 2\boldsymbol{\theta}^\top X^\top\mathbf{y} + \mathbf{y}^\top\mathbf{y}\bigr)$
    
    - 二次项：$\frac{1}{2m},\boldsymbol{\theta}^\top X^\top X\boldsymbol{\theta}$
        
    - 线性项：$-\frac{1}{m},\boldsymbol{\theta}^\top X^\top \mathbf{y}$
        
    - 常数项：$\frac{1}{2m},\mathbf{y}^\top \mathbf{y}$（对 $\theta$ 无影响）
        

---

## 4. 梯度推导

- **二次型梯度**：对 $f(\boldsymbol{\theta}) = \boldsymbol{\theta}^\top A\boldsymbol{\theta}, A$ 对称  
    $\nabla_\theta f = 2A\boldsymbol{\theta}$
    
- **具体到线性回归**：  
    $\nabla J(\boldsymbol{\theta}) = \frac{1}{2m} \nabla\bigl(\boldsymbol{\theta}^\top X^\top X\boldsymbol{\theta} - 2\boldsymbol{\theta}^\top X^\top\mathbf{y} + \mathbf{y}^\top\mathbf{y}\bigr) = \frac{1}{m}\bigl(X^\top X\boldsymbol{\theta} - X^\top\mathbf{y}\bigr)$
    
- **批量梯度表达**：  
    $\nabla J = \frac{1}{m} X^\top (X,\boldsymbol{\theta} - \mathbf{y})$
    

---

## 5. 梯度下降更新规则

- **更新公式**：  
    $\boldsymbol{\theta} := \boldsymbol{\theta} - \alpha,\nabla J(\boldsymbol{\theta})$  
    其中 $\alpha$ 是学习率。
    
- **等价写法**（若损失定义为 $\frac{1}{m}\sum (\cdot)^2$）：  
    $\boldsymbol{\theta} := \boldsymbol{\theta} - \frac{2\alpha}{m}X^\top(X\boldsymbol{\theta}-\mathbf{y})$
    

---

## 6. 广义多元线性回归

- **特征扩展**：可支持任意维度特征；按相同流程构造 $X, \boldsymbol{\theta}$。
    
- **目标与梯度**：  
    $J = \frac{1}{2m}\lVert X\boldsymbol{\theta}-\mathbf{y}\rVert^2, \quad \nabla J = \frac{1}{m}X^\top(X\boldsymbol{\theta}-\mathbf{y})$
    
- **正则化扩展**：可在损失中加入 $L1$ 或 $L2$ 正则项，控制模型复杂度。
## 🛡️ 二、线性回归的正则化扩展

**为什么需要正则化？**  
当特征很多或者存在**共线性**时，模型可能**过拟合**。  
为了解决这个问题，我们可以在损失函数中加入**正则项**来惩罚参数过大。

---

### 🔹 1. L2 正则化（Ridge 回归）

在原损失函数上加上所有参数的平方和：

$J_{\text{ridge}}(\boldsymbol{\theta}) = \frac{1}{2m} \lVert X\boldsymbol{\theta} - \mathbf{y} \rVert^2 + \frac{\lambda}{2m} \lVert \boldsymbol{\theta}_{1:} \rVert^2$

- $\lambda$：正则化强度（超参数）  
- $\boldsymbol{\theta}_{1:}$：表示除偏置 $\theta_0$ 外的其余参数

**梯度变为：**

$\nabla J_{\text{ridge}} = \frac{1}{m} X^\top (X\boldsymbol{\theta} - \mathbf{y}) + \frac{\lambda}{m} \boldsymbol{\theta}_{1:}$

> 🚀 Ridge 会抑制参数过大，提升模型泛化能力。

---

### 🔹 2. L1 正则化（Lasso 回归）

在损失函数中加上参数绝对值的和：

$J_{\text{lasso}}(\boldsymbol{\theta}) = \frac{1}{2m} \lVert X\boldsymbol{\theta} - \mathbf{y} \rVert^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$

- **Lasso 优势**：可以产生稀疏解（即某些 $\theta_j = 0$），实现特征选择。  
- **劣势**：绝对值不可导，不能用标准梯度下降求解，需用坐标下降、子梯度法等。

---

### 🔹 3. 弹性网（Elastic Net）

将 L1 和 L2 结合：

$J_{\text{elastic}} = \frac{1}{2m} \lVert X\boldsymbol{\theta} - \mathbf{y} \rVert^2 + \frac{\lambda_1}{m} \lVert \boldsymbol{\theta} \rVert_1 + \frac{\lambda_2}{2m} \lVert \boldsymbol{\theta} \rVert_2^2$

适合处理特征高度相关的数据集。它同时具备 Lasso 的稀疏性与 Ridge 的稳定性。

---

### ✅ 正则化方法对比总结

| 方法          | 正则项形式            | 优点             | 缺点        |
| ----------- | ---------------- | -------------- | --------- |
| 无正则         | 无                | 简单、直接          | 容易过拟合     |
| Ridge (L2)  | $\|\theta\|_2^2$ | 稳定，防止过拟合       | 不会产生稀疏解   |
| Lasso (L1)  | $\|\theta\|_1$   | 稀疏性，特征选择       | 解不稳定，优化复杂 |
| Elastic Net | L1 + L2          | 综合 L1 和 L2 的优点 | 参数多，调参更复杂 |
