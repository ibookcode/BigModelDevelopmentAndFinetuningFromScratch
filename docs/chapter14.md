# 第14章 ChatGPT前身——只具有解码器的GPT-2模型

## 14.1 GPT-2模型简介

### 14.1.2 GPT-2模型的PyTorch实现

[_14_1_2.py](../src/chapter14/_14_1_2.py)

[gpt2_config.py](../src/chapter14/gpt2_config.py)

### 14.1.3 GPT-2模型输入输出格式的实现

[get_data_emotion.py](../src/chapter14/get_data_emotion.py)

## 14.2 HuggingFace GPT-2模型源码模型详解

### 14.2.1 GPT2LMHeadModel类和GPT2Model类详解

[14_2_1.py](../src/chapter14/14_2_1.py)

### 14.2.2 Block类详解

[14_2_2.py](../src/chapter14/14_2_2.py)

### 14.2.3 Attention类详解

[14_2_3.py](../src/chapter14/14_2_3.py)

### 14.2.4 MLP类详解

[14_2_4.py](../src/chapter14/14_2_4.py)

## 14.3 HuggingFace GPT-2模型的使用与自定义微调

### 14.3.1 模型的使用与自定义数据集的微调

[14_3_1.py](../src/chapter14/14_3_1.py)

### 14.3.2 基于预训练模型的评论描述微调

[14_3_2.py](../src/chapter14/14_3_2.py)
