import json

import openai
import numpy as np
import pandas as pd
from utils.utils import get_project_root
# from config import OPENAI_KEY
# openai.api_key = OPENAI_KEY
import os
class Ada_Embedding_Model():

    def __init__(self):
        openai.api_key = ""
        openai.api_base = ""

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """ 获取单个文本的向量表示 """
        try_times = 0
        max_try_times = 5
        while try_times < max_try_times:
            try:
                res = openai.Embedding.create(input=[text], model=model)
                return res.data[0].embedding
            except Exception as e:
                print(f"Error: {e}")
                try_times += 1

        raise Exception("Failed to get embedding after 5 attempts")

    # text 为一个列表
    def get_embeddings(self, text, model="text-embedding-ada-002"):
        """ 获取多个文本的向量表示 """

        res = openai.Embedding.create(input=text, model=model)

        result = []
        for data in res.data:
            # result.append({
            #     "sentence": text[data.index],
            #     "embedding": data.embedding,
            # })
            result.append(data.embedding)
        return result, res.usage.total_tokens

    def cosine_similarity(self, a, b):
        """ 计算余弦相似度 """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
