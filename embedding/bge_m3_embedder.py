from FlagEmbedding import BGEM3FlagModel

from pathlib import Path
LOCAL_MODEL_PATH = Path(__file__).resolve().parent / "bge-m3"

from embedding.base_embedder import BaseEmbedder


class BgeM3Embedder(BaseEmbedder):
    def __init__(self, model_name_or_path=str(LOCAL_MODEL_PATH), use_fp16=True, batch_size=16, max_length=1024):
        self.model = BGEM3FlagModel(model_name_or_path, use_fp16=use_fp16, devices="cuda:0")
        self.batch_size = batch_size
        self.max_length = max_length
        self.dense_dim = 1024
        self.has_sparse = True

    # embed_texts 方法保持不变

    def embed_texts(self, text_list):
        # 同时算稠密和稀疏向量，不要 colbert 多向量
        output = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vector_list = output["dense_vecs"].tolist()

        # 稀疏向量：{词id: 权重}，词id 转成 int，Milvus 要求键是整数
        sparse_vector_list = []
        for token_weight_dict in output["lexical_weights"]:
            sparse_vector = {}
            for token_id, weight in token_weight_dict.items():
                sparse_vector[int(token_id)] = float(weight)
            sparse_vector_list.append(sparse_vector)

        return dense_vector_list, sparse_vector_list

if __name__ == "__main__":
    embedder = BgeM3Embedder()

    text_list = ["FTP 默认端口 21", "SSH 默认端口 22"]
    dense_vector_list, sparse_vector_list = embedder.embed_texts(text_list)

    # 稠密向量：应该是 2 条，每条 1024 维
    print("稠密向量条数:", len(dense_vector_list))
    print("稠密向量维度:", len(dense_vector_list[0]))

    # 稀疏向量：{词id: 权重}，打印第一条的词数和前 3 项
    print("第一条稀疏向量词数:", len(sparse_vector_list[0]))
    print("第一条稀疏向量前3项:", list(sparse_vector_list[0].items())[:3])

