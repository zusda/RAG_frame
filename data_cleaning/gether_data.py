from pathlib import Path

from data_cleaning.jsonl_io import save_documents_to_jsonl

INPUT_FOLDER = r"D:\1\信息收集--常见端口及服务_TXT"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "documents.jsonl"


def read_text_file(file_path, encoding_list):
    # 按顺序尝试每种编码，第一个读成功的就返回正文
    for encoding_name in encoding_list:
        try:
            return file_path.read_text(encoding=encoding_name)
        except UnicodeDecodeError:
            continue
    # 所有编码都失败，返回 None 交给调用方记录
    return None


def load_txt_documents(folder_path, recursive=True, encoding_list=("utf-8-sig", "gbk")):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"路径不是文件夹: {folder_path}")

    # 排序保证每次运行顺序一致
    if recursive:
        file_path_list = sorted(folder.rglob("*.txt"))
    else:
        file_path_list = sorted(folder.glob("*.txt"))

    document_list = []
    failed_path_list = []
    for file_path in file_path_list:
        try:
            content = read_text_file(file_path, encoding_list)
        except OSError as error:
            print(f"读取失败（IO错误）: {file_path} {error}")
            failed_path_list.append(str(file_path))
            continue

        if content is None:
            print(f"读取失败（编码无法识别）: {file_path}")
            failed_path_list.append(str(file_path))
            continue

        relative_path = file_path.relative_to(folder)
        title = relative_path.with_suffix("").as_posix()
        document_list.append({
            "metadata": {"name": relative_path.as_posix()},
            "content": f"标题：{title}\n\n{content}",
        })

    return document_list, failed_path_list


def gather_data(input_folder, output_path):
    # 读入 -> 整理成 document_list -> 保存成 jsonl
    document_list, failed_path_list = load_txt_documents(input_folder)
    print(f"成功读入 {len(document_list)} 个文件，失败 {len(failed_path_list)} 个")

    save_documents_to_jsonl(document_list, output_path)
    print(f"已保存到: {output_path}")
    return document_list, failed_path_list


if __name__ == "__main__":
    gather_data(INPUT_FOLDER, OUTPUT_PATH)