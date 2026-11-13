


from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    def __init__(self, embedder):
        # 向量库持有 embedding 模型，保证入库和查询用同一个模型
        self.embedder = embedder

    @abstractmethod
    def create_collection(self, collection_name, recreate=False):
        # 创建一份独立的数据集合；向量维度从 self.embedder.dense_dim 取
        # 已存在时，recreate=False 直接跳过，True 则删掉重建
        pass

    @abstractmethod
    def insert(self, collection_name, document_list):
        # document_list 每个元素：{"metadata": {...}, "content": "正文"}，就是前面整理好的格式
        # 子类内部调用 self.embedder.embed_texts 算出向量再入库
        pass

    @abstractmethod
    def search(self, collection_name, query_text, top_k=5):
        # 子类内部把 query_text 向量化后检索；filter_dict 是中立的等值过滤，None 表示不过滤
        # 返回 list，按相关度从高到低，每个元素：{"content": "正文", "metadata": {...}, "score": 相关度}
        pass


    @abstractmethod
    def load_collection(self, collection_name):
        # 打开已有的库之后、检索之前调用一次，把集合准备到可检索状态
        # 不需要这一步的后端（比如不分加载状态的）直接 pass
        pass

