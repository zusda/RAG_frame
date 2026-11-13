from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    # 稠密向量维度，子类必须在 __init__ 里设置，建集合时要用
    dense_dim = None
    has_sparse = False

    @abstractmethod
    def embed_texts(self, text_list):
        # 返回 (dense_vector_list, sparse_vector_list)
        # dense_vector_list：每条文本一个 list[float]
        # sparse_vector_list：每条文本一个 {词id: 权重}；模型不支持稀疏向量时返回 None
        pass