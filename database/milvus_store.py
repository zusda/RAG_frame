



import json

from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

from database.base_store import BaseVectorStore


class MilvusVectorStore(BaseVectorStore):
    def __init__(self, embedder, uri):
        super().__init__(embedder)
        # uri 传本地 .db 文件路径就是 Milvus Lite，传服务地址就是完整版
        self.client = MilvusClient(uri)
        # 如果传的是本地 .db 文件路径
        # 1、文件存在：打开它，里面的集合和数据都在。
        # 2、文件不存在：新建一个空库文件，此时没有任何集合，直接检索会报集合不存在。
        #
        # 如果传的是服务地址（如 http://localhost:19530），则是连接远程服务，不涉及文件创建。

    def create_collection(self, collection_name, recreate=False):
        if self.client.has_collection(collection_name):
            if not recreate:
                return
            self.client.drop_collection(collection_name)


        # 建 schema 时写了 auto_id=True。主键 id 会自动生成，无需自己插入。
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)

        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.embedder.dense_dim)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.JSON)

        # 稠密用余弦相似度
        index_params = self.client.prepare_index_params()   # 创建一个空的"索引参数容器"

        # 往 index_params 里登记一条规则，意思是"给 dense_vector 这个字段建一个 FLAT 索引，相似度用余弦"。
        # 此时只是登记，还没有真正建索引，要到 create_collection 时才会创建。
        index_params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="COSINE")
        # 索引是一种为了加快查找而提前建好的数据结构。有索引时，数据库提前把向量组织成某种结构，查询时只需要和一小部分向量比较，速度快很多，代价是结果变成"近似"，可能漏掉个别真正最相似的向量。
        # FLAT（暴力检索），把查询向量和库里每一个向量逐个算相似度，再排序。结果最精确，但数据量越大越慢。


        # 只有 embedder 支持稀疏向量时，才建稀疏字段和索引
        if self.embedder.has_sparse:
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

        self.client.create_collection(
            collection_name=collection_name,    # 集合名
            schema=schema,              # 字段
            index_params=index_params,  # 索引
        )

    def insert(self, collection_name, document_list):
        content_list = []
        for document in document_list:
            content_list.append(document["content"])

        # 向量化；sparse_vector_list 在模型不支持稀疏向量时是 None
        # 用embedding制作 稠密向量 和 稀疏向量
        # bge-m3 一次前向计算同时给出稠密和稀疏两种向量，所以 embed_texts 一次调用就返回两个 list
        dense_vector_list, sparse_vector_list = self.embedder.embed_texts(content_list)
        # 1、同一个模型（XLM-RoBERTa 架构，约 568M 参数）对输入文本做一次前向计算，得到每个 token 位置的隐藏状态。
        # 2、稠密向量取的是 [CLS] 位置的隐藏状态，归一化后就是 1024 维向量。
        # 3、稀疏向量是在最后一层隐藏状态上接一个小的线性层，对每个 token 输出一个非负权重，同一个词出现多次时取最大权重，再去掉特殊符号，得到 {词id: 权重}。这个线性层的参数是训练时学出来的。

        # {词id: 权重}，这是稀疏向量的一种存储写法。只记录非零的那几维：键是第几维（词 id），值是这一维的取值（权重）。没写出来的维度默认都是 0。

        # 1、稠密向量：做语义相似性检索，找"意思相近"的文本，即使字面不同。问"远程桌面用什么端口"，能找到讲 RDP 的块，即使块里没有"远程桌面"这四个字。
        # 2、稀疏向量：做关键词匹配，找"包含相同词"的文本，只认字面。问"3389"，只有真的含有"3389"的块得分高。


        # 制作 与schema相同字段的dict数据
        record_list = []
        for index, document in enumerate(document_list):
            record = {
                "dense_vector": dense_vector_list[index],
                "content": document["content"],
                "metadata": document["metadata"],
            }
            if self.embedder.has_sparse:
                record["sparse_vector"] = sparse_vector_list[index]
            record_list.append(record)

        # 分批插入，避免一次性数据过大
        insert_batch_size = 256
        for start in range(0, len(record_list), insert_batch_size):
            self.client.insert(
                collection_name=collection_name,
                data=record_list[start:start + insert_batch_size],
            )

    def build_filter_expr(self, filter_dict):
        # 把 {"name": "21.txt"} 翻译成 metadata["name"] == "21.txt"，多个条件用 and 连接
        if not filter_dict:
            return ""
        condition_list = []
        for key, value in filter_dict.items():
            condition_list.append(f'metadata["{key}"] == {json.dumps(value, ensure_ascii=False)}')
        return " and ".join(condition_list)


    def search(self, collection_name, query_text, top_k=5):
        dense_query_list, sparse_query_list = self.embedder.embed_texts([query_text])
        dense_query_vector = dense_query_list[0]
        output_field_list = ["content", "metadata"]

        if self.embedder.has_sparse:
            # 稠密和稀疏各检索一次，再用 RRF（按名次融合）合并
            dense_request = AnnSearchRequest(
                data=[dense_query_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE"},
                limit=top_k,
            )
            sparse_request = AnnSearchRequest(
                data=[sparse_query_list[0]],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=top_k,
            )
            raw_result_list = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=[dense_request, sparse_request],
                ranker=RRFRanker(60),
                limit=top_k,
                output_fields=output_field_list,
            )
        else:
            raw_result_list = self.client.search(
                collection_name=collection_name,
                data=[dense_query_vector],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE"},
                limit=top_k,
                output_fields=output_field_list,
            )

        # 只有一个查询，取第一组命中结果，整理成基类约定的格式
        hit_list = []
        for hit in raw_result_list[0]:
            hit_list.append({
                "content": hit["entity"]["content"],
                "metadata": hit["entity"]["metadata"],
                "score": hit["distance"],
            })
        return hit_list

    def load_collection(self, collection_name):
        # 重新打开已有的 db 文件时，集合处于未加载状态，必须先加载才能检索
        self.client.load_collection(collection_name)

if __name__ == "__main__":
    from pathlib import Path

    from embedding.bge_m3_embedder import BgeM3Embedder

    embedder = BgeM3Embedder()
    db_path = Path(__file__).resolve().parent / "milvus_test.db"
    store = MilvusVectorStore(embedder, str(db_path))

    collection_name = "test_collection"
    # recreate=True：每次运行都重建，保证测试环境干净
    store.create_collection(collection_name, recreate=True)

    # 用前面整理好的块格式造 3 条测试数据
    test_document_list = [
        {"metadata": {"name": "21.txt", "id": 0, "chunk_index": 0},
         "content": "标题：21\n\nFTP 默认端口 21，用于文件传输。"},
        {"metadata": {"name": "22.txt", "id": 1, "chunk_index": 0},
         "content": "标题：22\n\nSSH 默认端口 22，用于远程登录。"},
        {"metadata": {"name": "3389.txt", "id": 2, "chunk_index": 0},
         "content": "标题：3389\n\nRDP 默认端口 3389，Windows 远程桌面使用。"},
    ]
    store.insert(collection_name, test_document_list)

    # 测试 1：普通检索，预期第一条是 21.txt
    print("=== 测试1：FTP 用什么端口 ===")
    for hit in store.search(collection_name, "FTP 用什么端口", top_k=3):
        print(hit["metadata"]["name"], round(hit["score"], 4))

    # 测试 2：带过滤，只在 22.txt 里检索，结果只能有 22.txt
    print("=== 测试2：过滤 name=22.txt ===")
    for hit in store.search(collection_name, "远程登录", top_k=3):
        print(hit["metadata"]["name"], round(hit["score"], 4))

