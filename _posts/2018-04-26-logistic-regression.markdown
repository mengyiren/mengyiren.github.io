---
title: 吴恩达机器学习-逻辑回归
tags: 机器学习 逻辑回归 梯度下降 logistic_regression
---
## 1. 确定模型函数
首先明确逻辑回归的问题归属，逻辑回归属于分类问题，逻辑回归的分布属于多项式分布（包括二项分布，也叫伯努利分布）。这个笔记我们讨论二项分布的分类问题，通俗地讲二项分布是指我们要研究的问题只有两个结果，通常这两个结果为：是和否。我们一般喜欢用0和1分别代表否和是，那么什么样的函数能描述这样一种模型呢？这种函数模型叫sigmoid函数。  

函数如下图所示：
![sigmoid函数]({{ site.baseurl }}/assets/img/logistic_regression/sigmoid函数.png)  
函数的公式为$$h_\theta(x)=\frac{1}{1+e^{-x}}$$
## 2. 确定代价函数
首先，我们假设分类结果为1的概率为$$p(y=1|x;\theta)=h_\theta(x)$$  
那么结果为0的概率为$$p(y=0|x;\theta)=1-h_\theta(x)$$  
将上述两个概率写为一个公式$$p(y|x)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}$$  
假设输入特征值是独立随机分布的，那么我们可以通过计算极大似然值来确定当输入取x时，y的最可能取值   
$$L(\theta)=p(y|X;\theta)$$

$$=\prod_{i=0}^np(y^{(i)}|x^{(i)};\theta)$$

$$=\prod_{i=0}^n(h_\theta(x^{(i)}))^{y^{i}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}$$

为了简化计算我们同时对两个式子取对数   
$$J(\theta)=logL(\theta)$$

$$=\sum_{i=0}^ny^{(i)}logh(x^{(i)})+(1-y^{(i)})log(1-h(x^{(i)}))$$

以上是我们推导出代价函数的过程
## 3. 确定优化算法
我们在第二步计算出了代价函数，我们希望代价函数越大越好，借鉴梯度下降算法，我们可以给出参数的就计算公式：
$$\theta:=\theta+\alpha\Delta_\theta J(\theta)$$   

下面给出代价函数的偏导数的证明过程
![sigmoid函数]({{ site.baseurl }}/assets/img/logistic_regression/代价函数偏导数.png)  
将偏导数带入到第一个公式，得出我们的梯度上升公式  
$$\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}$$  
在这里提出一个小问题，线性回归使用的梯度下降算法跟逻辑回归使用的梯度上升算法是同一个吗？

除了使用梯度上升算法来求解参数外，我们还可以适用牛顿算法来求解参数，牛顿法的核心是要求出一个Heassian矩阵，矩阵中每个元素的值为 
![sigmoid函数]({{ site.baseurl }}/assets/img/logistic_regression/hessian矩阵元素.png)  
可以从上面公式得出Heassian矩阵为代价函数矩阵的二阶导数

## 4. python代码实现
上述两个公式的矩阵推导过程就不再证明，基础的矩阵推导知识可以查看上一章线性回归相关内容    
梯度上升矩阵计算公式：   
![sigmoid函数]({{ site.baseurl }}/assets/img/logistic_regression/梯度上升算法.png)

牛顿方法的推导过程：  
![sigmoid函数]({{ site.baseurl }}/assets/img/logistic_regression/牛顿方法推导.png)


```python
  # 梯度上升算法计算参数
  # alpha 步长 
def gradient_ascent(x, y, alpha):
    x = np.mat(x)
    y = np.mat(y)
    m, n = np.shape(x)
    weigh = np.ones((n, 1))
	grad = np.ones((n,1))
    while la.norm(grad) > 10E-6:
        h = sigmoid(x * weigh)
		grad = x.transpose() * (y - h)
        # 核心梯度上升算法
        weigh = weigh + alpha * grad
    return weigh
```

```python
  def newton_method(x, y, max):
    x = mat(x)
    y = mat(y)
    m, n = shape(x)
    weigh = zeros((n, 1))
    while la.norm(grad) > 10E-6:
        # 计算假设函数，得到一个列向量，每行为那个样本属于1的概率
        h = sigmoid(x * weigh)
        # 计算J对theta的一阶导数
        grad = x.transpose() * (y - h)
        # 计算海森矩阵即J对theta的二阶导数
        H = x.T * diagflat(h) * diagflat(h - 1) * x
        # 迭代求出theta
        weigh = weigh - la.inv(H) * grad
        plt.figure(1)
        plt.subplot(211)
        plt.plot(sort(x * weigh).tolist(), y.tolist(), 'b', label='real')

        plt.subplot(212)
        plt.plot(sort(x * weigh).tolist(), (2 * sigmoid(x * weigh) - 1).tolist(), 'r--', label='prediction')
        plt.show()
        time.sleep(5)
        max -= 1
    return weigh
```

## 5. 扩展

牛顿算法是二次收敛的，举个例子说明二次收敛。如果第一次迭代精度是0.1，那么第二次迭代的精度是0.01，第三次迭代的精度是0.0001...以此类推，因此牛顿算法的收敛性比梯度下降好很多，但是当训练样本数量非常多时，计算hesssian矩阵会消耗大量资源，可能造成内存不够用的情况。对于大多数分布式计算梯度上升算法用的是比较多的

以上证明了当结果为0，1时的逻辑回归的模型，代价函数，以及python程序，你能自己写出当结果为1，-1时逻辑回归的模型、代价函数，并且优化算法吗？如果你没有灵感，可以去我的[GitHub][1]寻找一下灵感

[1]:https://github.com/mengyiren/MachineLearning

