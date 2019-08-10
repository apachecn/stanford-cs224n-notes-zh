# Lecture 11: NLP中的卷积神经网络
## 从循环神经网络到卷积神经网络
循环神经网络不能捕获没有前文的短语

通常在最后的向量中捕获太多最后的单词
- E.g., softmax通常在最后一步使用

卷积层的**主要思想**：
如果我们对于每个特定长度的所有可能的单词子序列都计算向量呢？
- 举个例子: 对"tentative deal reached to keep government open" 这句话会计算的短语向量如下:
- - tentative deal reached, deal reached to, reached to keep, to keep government, keep government open

不在乎短语是否合乎语法

在语言学或认知学上都不太可信

> 有关卷积在其他领域的含义以及它在图像领域的研究在此不展开，有兴趣的自行搜索

## 卷积神经网络在NLP中的应用
下面将以上面提到的英文句子为例，直观地看一些基本的、常见的用于文本的卷积操作
#### a) 用于文本处理的一维卷积
![pic4](https://github.com/Originval/Learning/blob/master/pics/pic4.png?raw=true)

#### b) 带有填充(padding)的用于文本处理的一维卷积
![pic5](https://github.com/Originval/Learning/blob/master/pics/pic5.png?raw=true)

#### c) 通道数为3，padding为1的一维卷积
![pic6](https://github.com/Originval/Learning/blob/master/pics/pic6.png?raw=true)

#### d) 一维卷积，带填充和基于时间的最大池化
![pic7](https://github.com/Originval/Learning/blob/master/pics/pic7.png?raw=true)

#### e) 一维卷积，带填充和基于时间的平均池化
![pic8](https://github.com/Originval/Learning/blob/master/pics/pic8.png?raw=true)

### 使用PyTorch的相关参数

```
batch_size = 16
word_embed_size = 4
seq_len = 7
input = torch.randn(batch_size, word_embed_size, seq_len)
conv1 = Conv1d(in_channels=word_embed_size, out_channels=3, kernel_size=3)   # can add: padding=1
hidden1 = conv1(input)
hidden2 = torch.max(hidden1, dim=2)     # max pool
```

## 用于句子分类的单层卷积神经网络
参考论文：
1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) Yoon Kim (2014)  EMNLP. 
2. A variant of convolutional NNs of Collobert, Weston et al. (2011)

目标：句子分类
- 主要对句子是积极的还是消极的进行情感分类
- 其他的任务有：
- - 主语和客观语句的分类
- - 对有关任务，地点，数字等问题的分类

> 其他的一些符号或说明可查看相关的论文

### 单层卷积神经网络
过滤器w会被用到所有可能的窗口上 (连接的向量)

卷积层中计算特征（单通道）的公式：![image](https://github.com/Originval/Learning/blob/master/pics/1.png?raw=true)

句子的表示形式：![image](https://github.com/Originval/Learning/blob/master/pics/2.png?raw=true)

所有可能的窗口长度为h：![image](https://github.com/Originval/Learning/blob/master/pics/3.png?raw=true)

返回结果是一个特征映射：![image](https://github.com/Originval/Learning/blob/master/pics/4.png?raw=true)

### 池化和通道
池化采用的是基于时间的最大池化

思路：捕获最重要的激活值

通过特征映射 ![image](https://github.com/Originval/Learning/blob/master/pics/4.png?raw=true)

经池化得到的单个数字：![image](https://github.com/Originval/Learning/blob/master/pics/6.png?raw=true)

使用的多个过滤器的权重为w，采用不同的窗口尺寸h会很有用，因为最大池化![image](https://github.com/Originval/Learning/blob/master/pics/6.png?raw=true)，c的无关的长度![image](https://github.com/Originval/Learning/blob/master/pics/4.png?raw=true)，所以我们可以用一些过滤器来观察1元语法，2元语法，3元语法，等等。

### 多通道输入的思路
- 用预训练词向量初始化（word2vec或Glove）
- 从两份副本开始
- 只通过其中一个集合反向传播，其他保持静态
- 在最大池化前，把两个通道集都加到ci

### 在一层卷积层之后分类
![(p18)](https://github.com/Originval/Learning/blob/master/pics/pic18.png?raw=true)
- 首先是一个卷积层，然后是最大池化层
- 获取最终的特征向量（使用100个特征映射，每个尺寸是3，4，5）
- 最后是简单的softmax层

参考论文：[Zhang and Wallace(2015) A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf)

### 正则化
使用Dropout：为概率p（超参数）为1的伯努利随机变量创建mask向量r；在训练过程中删除一些特征：![image](https://github.com/Originval/Learning/blob/master/pics/7.png?raw=true)。这样做是为了防止共适应（对于特定特征群的过度拟合）。但是在进行测试时，不需要dropout，按照概率p缩放最终向量![image](https://github.com/Originval/Learning/blob/master/pics/8.png?raw=true)

也可以对每个类的权重向量（softmax权重中的每行）使用l2正则化约束为固定值s（也是超参数）。如果![image](https://github.com/Originval/Learning/blob/master/pics/9.png?raw=true)，就重新缩放![image](https://github.com/Originval/Learning/blob/master/pics/10.png?raw=true)

### 模型比较
- Bag of Vectors: 对简单的分类问题而言是很好的基准模型。尤其在其后加上ReLU层
- Window Model：对不需要广泛上下文的单个词语的分类问题很不错。例如：词性标注，命名实体识别
- CNNs： 分类的效果很好，对一些较短的短语需要0填充，较难解释，但比较容易用GPU并行训练。高效且功能丰富强大
- Recurrent Neural Networks：在认知上似是而非，不适合分类（如果只使用最后的状态），比卷积神经网络慢很多，适合序列标记和分类，对语言模型很好，与注意力机制结合会很出色

### 门控单元的垂直使用
![(p45)](https://github.com/Originval/Learning/blob/master/pics/45.png?raw=true)
- 我们在LSTMs和GRUs里看到的门控/跳过是一个总体概念，现在在很多地方使用
- 关键思想——用捷径连接对候选更新进行求和，是让很深的网络工作所必须的

### 批归一化(BatchNorm)
- 通过使用卷积神经网络
- 通过缩放激活值使其均值和单位方差为0来转换批次的卷积输出
- - 这是统计学中常见的z变换
- - 但是每批次更新一次，所以波动不会影响太多
- 使用批归一化使模型对参数初始化不那么敏感，因为输出使自动重新标度的
- - 这也使对学习率的微调更简单
- 在PyTorch中使用nn.BatchNorm1d

### 1x1卷积
1x1卷积，即网络中的网络(Network-in-Network)连接，具有kernel_size=1的卷积内核,它给你跨越通道的全连接线性层，也可用于从多个通道映射到更少的通道。

此外，1x1卷积只增加了很少的额外参数，就增加了额外的神经网络层，不像全连接(FC)那样增加了很多参数。

### CNN的应用：翻译
![(p44)](https://github.com/Originval/Learning/blob/master/pics/44.png?raw=true)

- 最早成功的神经机器翻译之一
- 使用CNN进行编码，RNN进行解码

#### 论文 Learning Character-level Representations for Part-of-Speech Tagging
- 对字符的卷积生成词嵌入
- 使用PoS标记的固定词嵌入窗口长度

#### 论文 Character-Aware Neural Language Models
- 基于字符的词嵌入
- 利用卷积，Highway网络和LSTM

## Quasi-Recurrent 神经网络
![(p42)](https://github.com/Originval/Learning/blob/master/pics/42.png?raw=true)
- 尝试将LSTM和CNN这两个模型的优点结合在一起 
- 跨越时间的并行卷积
- 跨通道并行的元素门控伪递归是在池化层中完成的

### Q-RNN实验：语言模型
介绍了论文(ICLR 2017) Quasi Recurrent Neural Networks
![(p43)](https://github.com/Originval/Learning/blob/master/pics/43.png?raw=true)


### Q-RNNs在情感分析中的应用
通常比LSTM更好，更快

可解释性较好

### QRNN的局限性
1. 在字符级的语言模型中不能表现得像LSTM一样好
- 在更长的依赖中有困难

2. 通常需要更深的网络才能取得和LSTM一样的性能
- 当更深的时候它们仍然很快
- 它们用深度代替真正的循环很有效

### TransformersRNNs的缺点&Transformer的积极性
![(p41)](https://github.com/Originval/Learning/blob/master/pics/41.png?raw=true)
- 我们想要并行化，但是循环神经网络本质是顺序的
- 尽管有GRUs和LSTMs，循环神经网络仍然从处理长距离依赖的注意力机制中获益——状态之间的路径长度随顺序增长

