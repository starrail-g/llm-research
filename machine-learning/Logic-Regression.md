# 逻辑回归学习笔记

## 1. 概述  
- **定义**：逻辑回归（Logistic Regression）是一种**分类**算法，用于预测目标变量为离散类别（通常是二分类：0/1）的概率。它基于线性回归思想，加入了 Sigmoid 函数将输出“压缩”到 $(0,1)$ 区间。
- **应用场景**：医学诊断（良性/恶性）、信用评估（批准/拒绝）、营销响应（购买/不购买）等。

---

## 2. 模型形式  

1. **线性部分**  
   $$
   z = w^\top x + b
   $$
   - $x\in\mathbb{R}^n$：输入特征向量  
   - $w\in\mathbb{R}^n$，$b\in\mathbb{R}$：模型参数  

2. **激活函数——Sigmoid**  
   $$
   \sigma(z) = \frac{1}{1 + e^{-z}}
   $$
   - 性质：$\sigma(z)\to1\,(z\to+\infty)$，$\sigma(z)\to0\,(z\to-\infty)$，$\sigma(0)=0.5$  

3. **最终模型**  
   $$
   \hat y = f(x) = \sigma(w^\top x + b)
   $$
   - $\hat y\in(0,1)$：样本属于正类（$y=1$）的概率。

---

## 3. 预测与决策边界  
- **概率输出**：$\hat y=P(y=1\mid x)$。  
- **二值决策**：设阈值 $t$（常用 $t=0.5$），
  $$
  \begin{cases}
    \hat y \ge t\;\Rightarrow\;y_{\text{pred}}=1,\\
    \hat y < t\;\Rightarrow\;y_{\text{pred}}=0.
  \end{cases}
  $$  
- **决策边界**：$\{x\mid w^\top x + b = 0\}$，在二维时是一条直线；若引入多项式特征，则可得到圆、椭圆或更复杂曲线。

---

## 4. 参数估计：最大似然与损失函数  

### 4.1 最大似然估计（MLE）的思想  
- 目标：选择参数 $\theta=(w,b)$，使得在该参数下观测到训练数据的**似然**最大。  
- 对数似然（对所有样本）：
  $$
    \ell(w,b)
    = \sum_{i=1}^m \Bigl[y^{(i)}\log f(x^{(i)}) + (1-y^{(i)})\log\bigl(1-f(x^{(i)})\bigr)\Bigr]
  $$  
- 最大化 $\ell$ 等价于最小化负对数似然。

### 4.2 交叉熵损失（Log Loss）  
- **单样本损失**：
  $$
  L\bigl(f(x),y\bigr)
  = -\Bigl[y\log f(x) + (1-y)\log\bigl(1-f(x)\bigr)\Bigr]
  $$
  - $y\in\{0,1\}$：真实标签  
- **成本函数**（平均损失）：
  $$
  J(w,b)
  = \frac1m\sum_{i=1}^m L\bigl(f(x^{(i)}),y^{(i)}\bigr)
  = -\frac1m\sum_{i=1}^m\Bigl[y^{(i)}\log f(x^{(i)}) + (1-y^{(i)})\log\bigl(1-f(x^{(i)})\bigr)\Bigr].
  $$  
- **凸性保证**：该 $J(w,b)$ 是凸函数，梯度下降可稳定收敛到全局最优。

---

## 5. 模型训练：梯度下降  

1. **梯度**：
   $$
     \frac{\partial J}{\partial w} 
     = \frac1m \sum_{i=1}^m \bigl(f(x^{(i)}) - y^{(i)}\bigr)\,x^{(i)}, 
     \quad
     \frac{\partial J}{\partial b}
     = \frac1m \sum_{i=1}^m \bigl(f(x^{(i)}) - y^{(i)}\bigr).
   $$
2. **参数更新（批量梯度下降）**：
   $$
   w \gets w - \alpha\,\frac{\partial J}{\partial w}, 
   \quad
   b \gets b - \alpha\,\frac{\partial J}{\partial b}.
   $$
3. **优化器**：可用标准梯度下降、随机梯度下降（SGD）、带动量或自适应算法（Adam、L-BFGS 等）。

---

## 6. 正则化  

- 为防止过拟合，可在成本函数中加入正则项：  
  - **L2 正则**（Ridge）：
    $$
      J_{\text{reg}} = J + \frac{\lambda}{2m}\|w\|_2^2.
    $$
  - **L1 正则**（Lasso，需用特定优化器）  
- 在 scikit-learn 中，通过参数 `C=1/λ` 控制正则化强度，支持多种 solvers

---

## 7. 多分类扩展  
1. **一对多（OvR, One-vs-Rest）**：为每个类别训练一个二分类器。  
2. **Softmax 回归（Multinomial）**：直接训练多项逻辑模型，输出对各类别的概率。  
- 在 scikit-learn 中，`LogisticRegression(multi_class='multinomial', solver='lbfgs')`。

---

## 8. 模型评估指标  

- **混淆矩阵**：TP, FP, TN, FN  
- **准确率（Accuracy）**：$(\text{TP}+\text{TN})/m$  
- **精确率（Precision）**：$\text{TP}/(\text{TP}+\text{FP})$  
- **召回率（Recall）**：$\text{TP}/(\text{TP}+\text{FN})$  
- **F1 分数**：$2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$  
- **ROC 曲线 & AUC**：不同阈值下的 TPR vs FPR 曲线下面积。

---

## 9. 实践要点  

- **特征缩放**：对梯度方法和带正则化的模型尤为重要。  
- **类别编码**：One-Hot 或目标编码。  
- **学习率调度**：可动态调整 $\alpha$ 提升收敛速度。  
- **样本不平衡**：可调整阈值或采用类别权重（`class_weight` 参数）。  
- **超参数调优**：网格搜索（GridSearchCV）、随机搜索（RandomizedSearchCV）。

---

## 10. 优势与局限  

- **优势**：实现简单、训练速度快、输出概率可解释、模型可正则化。  
- **局限**：  
  - 只能学习由特征线性可分（或通过特征变换后可分）的决策边界；  
  - 对离群点敏感；  
  - 需要合理的特征工程才能获得好性能。

---

***  
## 公式推导

下面对笔记中出现的主要公式，给出逐步推导。

---



### 3. 成本函数（平均损失）推导

定义训练集 $\{(x^{(i)},y^{(i)})\}_{i=1}^m$。成本函数是所有样本的平均损失：
$$[
J(w,b)
= \frac1m \sum_{i=1}^m L\bigl(f(x^{(i)}),y^{(i)}\bigr)
= -\frac1m\sum_{i=1}^m\bigl[y^{(i)}\log f(x^{(i)}) + (1-y^{(i)})\log(1-f(x^{(i)}))\bigr].
$$

---

### 4. 梯度推导

我们要对 $J(w,b)$ 分别求关于 $w$ 和 $b$ 的偏导。记
$$[
z^{(i)} = w^\top x^{(i)} + b,\quad
\hat y^{(i)} = f(x^{(i)}) = \sigma\bigl(z^{(i)}\bigr).]$$

#### 4.1 对 $w$ 的导数

1. 写出 $J$：
   $$[
   J = -\frac1m\sum_{i=1}^m \bigl[y^{(i)}\log\hat y^{(i)} + (1-y^{(i)})\log(1-\hat y^{(i)})\bigr].
   $$]
2. 对单样本损失 $L^{(i)}$ 先对 $z^{(i)}$ 求导：
   $$
   \frac{\partial L^{(i)}}{\partial z^{(i)}}
   = -\Bigl[
     y^{(i)}\frac{1}{\hat y^{(i)}}\cdot\hat y^{(i)}(1-\hat y^{(i)})
     \;-\;(1-y^{(i)})\frac{1}{1-\hat y^{(i)}}\cdot\hat y^{(i)}(1-\hat y^{(i)})
     \Bigr].
   $$
   注意：$d\hat y/d z = \sigma(z)(1-\sigma(z))$。  
   化简得
  $$ 
   \frac{\partial L^{(i)}}{\partial z^{(i)}}
   = -\bigl[y^{(i)}(1-\hat y^{(i)}) - (1-y^{(i)})\hat y^{(i)}\bigr]
   = \hat y^{(i)} - y^{(i)}.
   $$
3. 再乘上 $z^{(i)}$ 对 $w$ 的导数 $\partial z^{(i)}/\partial w = x^{(i)}$：
  $$ 
   \frac{\partial L^{(i)}}{\partial w}
   = \bigl(\hat y^{(i)} - y^{(i)}\bigr)\,x^{(i)}.
  $$
4. 平均到所有样本：
$$   
   \frac{\partial J}{\partial w}
   = \frac1m \sum_{i=1}^m \frac{\partial L^{(i)}}{\partial w}
   = \frac1m \sum_{i=1}^m \bigl(\hat y^{(i)} - y^{(i)}\bigr)\,x^{(i)}.
   $$

#### 4.2 对 $b$ 的导数

同理，因 $\partial z^{(i)}/\partial b = 1$：
\[$$
\frac{\partial J}{\partial b}
= \frac1m \sum_{i=1}^m \bigl(\hat y^{(i)} - y^{(i)}\bigr).
$$

---

### 5. 参数更新（梯度下降）

将上述梯度带入更新公式：
\[$$
\begin{cases}
w \leftarrow w - \alpha\,\frac{\partial J}{\partial w},\\[5pt]
b \leftarrow b - \alpha\,\frac{\partial J}{\partial b}.
\end{cases}
$$
其中学习率 $\alpha>0$。

---


