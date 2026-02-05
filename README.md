# 电商评论情感分析数据处理项目 

## 项目简介 

基于Python+Pandas搭建电商评论数据处理工作流，解决原始数据重复/缺失/噪声问题，构建高质量情感分析训练数据集。 

数据集来源：[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download)

经历以下步骤来实现本项目：

1.加载原始数据：选取前10000行数据

2.数据探索（EDA）：观察数据特征

3.数据清洗：去重、处理缺失值、剔除无效文本、文本降噪

4.文本预处理 + 构建情感标签

5.保存

## 环境依赖 

- Python 3.8+ 

- pandas, re, numpy, nltk, openpyxl 

## 运行步骤 

1. 安装依赖：`pip install pandas numpy nltk openpyxl` 
2. 运行核心代码：`python processing.py` 
3.  查看最终成果：`cleaned_ecommerce_reviews.csv` 

## 核心成果 

1. 处理原始数据 10000 条，最终输出高质量数据 9513 条 ，正面评论占比 76.56%

2. 构建三元情感标签（正面/中性/负面） 

3. 输出数据集可直接用于情感分析模型训练