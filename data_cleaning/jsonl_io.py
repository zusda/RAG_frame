import json
from pathlib import Path


def save_documents_to_jsonl(document_list, output_path):
    output_file = Path(output_path)
    # 输出目录不存在就创建
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 显式指定 utf-8；每个文档写成一行 JSON，ensure_ascii=False 保留中文
    with output_file.open("w", encoding="utf-8") as file:
        for document in document_list:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def load_documents_from_jsonl(input_path):
    document_list = []
    with Path(input_path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            document_list.append(json.loads(line))
    return document_list


def save_dict_id2document(dict_id2document, output_path):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # JSON 的键只能是字符串，整数 id 写入时会自动变成 "0"、"1"
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(dict_id2document, file, ensure_ascii=False)


def load_dict_id2document(input_path):
    with Path(input_path).open("r", encoding="utf-8") as file:
        raw_dict = json.load(file)
    # 读回来后把字符串键转回整数，才能和块里 metadata["id"] 的整数对上
    dict_id2document = {}
    for doc_id_text, document in raw_dict.items():
        dict_id2document[int(doc_id_text)] = document
    return dict_id2document