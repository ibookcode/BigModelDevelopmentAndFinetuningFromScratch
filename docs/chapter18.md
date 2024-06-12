# 第18章 对训练成本上亿美元的ChatGLM进行高级微调

## 18.1 ChatGLM模型的本地化处理

### 18.1.1 下载ChatGLM源码与合并存档

[demo.py](../src/chapter18/chatGLM_spo/demo.py)

[modeling_chatglm.py](../src/chapter18/chatGLM_spo/huggingface_saver/modeling_chatglm.py)

[configuration_chatglm.py](../src/chapter18/chatGLM_spo/huggingface_saver/configuration_chatglm.py)

### 18.1.2 修正自定义的本地化模型

[xiaohua_model.py](../src/chapter18/chatGLM_spo/huggingface_saver/xiaohua_model.py)

## 18.2 高级微调方法1————基于加速库Accelerator的全量数据微调

### 18.2.1 数据的准备————将文本内容转化成三元组的知识图谱

[get_data.py](../src/chapter18/chatGLM_spo/get_data.py)

### 18.2.2 加速的秘密————Accelerate模型加速工具详解

[demo.py](../src/chapter18/chatGLM_spo/fintunning_PT/demo.py)

[pred_demo.py](../src/chapter18/chatGLM_spo/fintunning_PT/pred_demo.py)

### 18.2.3 更快的速度————使用INT8(INT4)量化模型加速训练

[quantization.py](../src/chapter18/chatGLM_spo/huggingface_saver/quantization.py)
