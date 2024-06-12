import numpy
from utils import mean_pooling,compute_sim_score


from transformers import BertTokenizer, BertModel
import torch
embedding_model_name = "shibing624/text2vec-base-chinese"
# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
model = BertModel.from_pretrained(embedding_model_name)

query = ["雅生活服务的人工成本占营业成本的比例是多少"]
# Tokenize sentences
# 对文本进行编码
query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
# Compute token embeddings
# 使用模型进行推断
with torch.no_grad():
    model_output = model(**query_input)
# Perform pooling. In this case, mean pooling.
# 对推断结果进行均值池化压缩
query_embedding = mean_pooling(model_output, query_input['attention_mask'])
print(query_embedding.shape)

import numpy as np
sentence_embeddings_np = np.load("./dataset/financial_research_reports/sentence_embeddings_np.npy")

# 将query的Embedding编码与上一步存档的Embedding编码进行相关性计算
#这里很容易认清楚是007
for i in range(len(sentence_embeddings_np)):
    # 计算余弦相似度
    score = compute_sim_score(sentence_embeddings_np[i],query_embedding[0])
    print(i,score)
