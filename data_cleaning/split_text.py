from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_cleaning.jsonl_io import load_documents_from_jsonl, save_documents_to_jsonl, save_dict_id2document

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
INPUT_PATH = OUTPUT_DIR / "documents.jsonl"
DOCUMENT_OUTPUT_PATH = OUTPUT_DIR / "documents_with_id.jsonl"
DICT_OUTPUT_PATH = OUTPUT_DIR / "dict_id2document.json"
CHUNK_OUTPUT_PATH = OUTPUT_DIR / "chunks.jsonl"

# 中文友好的分隔符，按优先级依次尝试：段落、换行、句末标点、逗号、空格，最后按单字符硬切
SEPARATOR_LIST = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def add_id_to_documents(document_list):
    # 第一个输出：原文档数据，元数据里加 id（按顺序从 0 开始），content 不变
    # 同时构造 dict_id2document：{id: 原文档}，检索到块后用 id 直接取回整篇原文
    document_with_id_list = []
    dict_id2document = {}
    for doc_id, document in enumerate(document_list):
        metadata = dict(document["metadata"])
        metadata["id"] = doc_id
        document_with_id = {
            "metadata": metadata,
            "content": document["content"],
        }
        document_with_id_list.append(document_with_id)
        dict_id2document[doc_id] = document_with_id
    return document_with_id_list, dict_id2document


def get_title_prefix(document):
    # 读入阶段在正文前拼了"标题：xxx\n\n"，这里重建出这个前缀；对不上就当作没有前缀
    title = Path(document["metadata"]["name"]).with_suffix("").as_posix()
    title_prefix = f"标题：{title}\n\n"
    if document["content"].startswith(title_prefix):
        return title_prefix
    return ""


def split_documents(document_with_id_list, chunk_size=500, chunk_overlap=50):
    # 第二个输出：切块后的数据，metadata 继承原文档的 name 和 id，再加 chunk_index
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATOR_LIST,
        keep_separator="end",  # 分隔符保留在块的末尾，避免块以"。"开头
    )

    chunk_list = []
    for document in document_with_id_list:
        # 先把开头的"标题：xxx\n\n"拆出来，只对纯正文切分
        title_prefix = get_title_prefix(document)
        body = document["content"][len(title_prefix):]

        for chunk_index, chunk_text in enumerate(text_splitter.split_text(body)):
            chunk_metadata = dict(document["metadata"])
            chunk_metadata["chunk_index"] = chunk_index
            chunk_list.append({
                "metadata": chunk_metadata,
                # 每块都重新拼上标题，保证每块都能靠文件名被检索到
                "content": title_prefix + chunk_text,
            })
    return chunk_list


def split_and_save(input_path, document_output_path, dict_output_path, chunk_output_path, chunk_size=500, chunk_overlap=50):
    document_list = load_documents_from_jsonl(input_path)
    document_with_id_list, dict_id2document = add_id_to_documents(document_list)
    # [
    #     {
    #         "metadata": {"name": "21.txt", "id": 0},
    #         "content": "标题：21\n\nFTP 默认端口 21，用于文件传输。",
    #     },
    #     {
    #         "metadata": {"name": "sub/22.txt", "id": 1},
    #         "content": "标题：sub/22\n\nSSH 默认端口 22，用于远程登录。……（较长的正文）",
    #     },
    # ]


    chunk_list = split_documents(document_with_id_list, chunk_size, chunk_overlap)
    # [
    #     {
    #         "metadata": {"name": "21.txt", "id": 0, "chunk_index": 0},
    #         "content": "标题：21\n\nFTP 默认端口 21，用于文件传输。",
    #     },
    #     {
    #         "metadata": {"name": "sub/22.txt", "id": 1, "chunk_index": 0},
    #         "content": "标题：sub/22\n\nSSH 默认端口 22，用于远程登录。……（前半段正文）",
    #     },
    #     {
    #         "metadata": {"name": "sub/22.txt", "id": 1, "chunk_index": 1},
    #         "content": "标题：sub/22\n\n……（后半段正文，可能和前一块有少量重叠）",
    #     },
    # ]

    save_documents_to_jsonl(document_with_id_list, document_output_path)
    save_dict_id2document(dict_id2document, dict_output_path)
    save_documents_to_jsonl(chunk_list, chunk_output_path)
    print(f"原文档 {len(document_with_id_list)} 个，切成 {len(chunk_list)} 块")
    return document_with_id_list, dict_id2document, chunk_list



if __name__ == "__main__":
    split_and_save(
        INPUT_PATH,
        DOCUMENT_OUTPUT_PATH,
        DICT_OUTPUT_PATH,
        CHUNK_OUTPUT_PATH,
        chunk_size=500,
        chunk_overlap=50,
    )