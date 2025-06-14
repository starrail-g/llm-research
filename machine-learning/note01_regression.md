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