# Notes of ***TensorFlow for Deep Learning***

## Charpter 4 - Fully Connected Deep Network

- 摈弃此前传统统计学中的**“正则”**概念，如LASSO等在神经网络中的用处可能没有想象中的大，深度神经网络中一般使用**Dropout/Early Stopping/Weight Regularization**等正则策略；

- Dropout主要是为了避免在神经网络训练过程中，深层结构过度依赖一个浅层神经元的特征或信息，导致网络过拟合，Dropout时并非是将该节点删除，而仅是置零；

- Dropout需要在进行训练时关闭；

- Early stopping是解决神经网络“记忆”或过拟合问题的另一种手段，是指**当验证集的准确率或其它衡量指标出现恶化**时即停止训练；

- 也可以使用参数L1或L2正则，但一般效果较前两者差；

## Charpter 5 - Hyperparameter Optimization

1. 超参数学习

    1. 超参数是指模型中不由Optimizer学习获得及更新的参数，比如隐层个数、隐层中神经元个数、学习率等；

    2. 可以考虑采用80/10/10的训练、验证、测试集划分，避免模型在验证集上过拟合；

    3. 一般认为训练集用于训练Gradient-based参数，验证集用于超参数训练，测试集用于检验超参数的泛化能力；

    4. 超参数学习过程是一个黑盒学习过程，由于缺乏方向信息，无法像白盒训练（如梯度下降）过程一样在高维空间泛化

2. 评价标准

    1. 二分类问题

       Precision = TP/(TP+FP)

       Recall = TP/(TP+FN) = TP/P

       Specificity = TN/(FP+TN) = TN/N

       FPR = FP/(FP+TN) = FP/P

       FNR = FN/(TP+FN) = FN/P

       TPR = TP/(TP+FN) = TP/P

       TNR = TN/(TN+FP) = TN/N

       以TPR为纵轴，FPR为横轴的曲线称为ROC曲线，下方面积为AUC

    2. 多分类问题

       使用准确率或混淆矩阵。

       3. 回归问题
          $$
          R = \frac{\sum_{i=1}^N(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^N(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^N(y_i-\hat{y})^2}}
          $$

          $$
          RMSE=\sqrt{\frac{\sum_{i=1}^N(x_i-y_i)^2}{N}}
          $$

3. 超参数优化方法
   1. Graduate Student Descent
   2. Grid Search
   3. Random Hyperparameter Search

## Charpter 6 - Convolutional Neural Networks







   