from pathlib import Path

from data_cleaning.gether_data import gather_data
from data_cleaning.split_text import split_and_save
from database.milvus_store import MilvusVectorStore

# 这个py用来创建4个json文件，以及1个db文件。


# 输出文件名固定，都放在 output_dir 下
DOCUMENTS_NAME = "documents.jsonl"
DOCUMENTS_WITH_ID_NAME = "documents_with_id.jsonl"
DICT_NAME = "dict_id2document.json"
CHUNKS_NAME = "chunks.jsonl"
DB_NAME = "port_service.db"
COLLECTION_NAME = "port_service"


def build_port_index(input_folder, output_dir, embedder, chunk_size=500, chunk_overlap=50):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / DOCUMENTS_NAME
    documents_with_id_path = output_dir / DOCUMENTS_WITH_ID_NAME
    dict_path = output_dir / DICT_NAME
    chunks_path = output_dir / CHUNKS_NAME
    db_path = output_dir / DB_NAME

    # 1、读入 txt，整理成 document_list，保存到 documents_path
    document_list, failed_path_list = gather_data(input_folder, documents_path)
    if not document_list:
        raise ValueError(f"没有读到任何 txt 文件: {input_folder}")

    # 2、加 id、切块，保存 documents_with_id、dict_id2document、chunks 三个文件
    document_with_id_list, dict_id2document, chunk_list = split_and_save(
        documents_path,
        documents_with_id_path,
        dict_path,
        chunks_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunk_list:
        raise ValueError("切分后没有任何块")

    # 3、建向量库并插入；recreate=True 会删掉同名旧集合，保证库里的数据和本次切分一致
    store = MilvusVectorStore(embedder, str(db_path))
    store.create_collection(COLLECTION_NAME, recreate=True)
    store.insert(COLLECTION_NAME, chunk_list)
    print(f"已入库 {len(chunk_list)} 个块: {db_path}，集合名 {COLLECTION_NAME}")

    return store, dict_id2document


if __name__ == "__main__":
    from embedding.bge_m3_embedder import BgeM3Embedder

    build_port_index(
        input_folder=r"D:\setool\1\信息收集--常见端口及服务_TXT",
        output_dir=Path(__file__).resolve().parent / "output",
        embedder=BgeM3Embedder(),
    )