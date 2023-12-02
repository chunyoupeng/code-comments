
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.document_loaders import PyPDFLoader
import os
import tiktoken  # !pip install tiktoken
import sys
import re
import os


def load_pdf(directory):
    """
    加载指定目录下的所有 PDF 文档。

    Args:
        directory: 包含 PDF 文档的目录路径。

    Returns:
        documents: 包含所有加载的文档的列表。
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"Begin loading {filename}")
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            document = loader.load()
            documents.extend(document)
            print(f"{filename} load successfully")
    return documents


def delete_space(path):
    """
    删除指定目录下所有文本文件中的空格和换行符。

    Args:
        path: 包含文本文件的目录路径。
    """
    import os

    folder_path = path

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    content = f.read()

                processed_content = content.replace(' ', '').replace('\n', '')

                with open(file_path, 'w') as f:
                    f.write(processed_content)

    print("处理完成!")


def tiktoken_len(text):
    """
    使用 tiktoken 计算文本的标记数量。

    Args:
        text: 要计算标记数量的文本。

    Returns:
        len(tokens): 文本的标记数量。
    """
    tokenizer = tiktoken.get_encoding('p50k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)



def clean_string(input_doc):
    """
    清理字符串的函数。

    这个函数用于清理输入文档中的字符串，包括去除多余的空格、换行符、特殊字符等。

    Args:
        input_doc: 输入的文档对象。

    Returns:
        清理后的文档对象。
    """
    placeholder = "<DOUBLE_NEWLINE>"
    step1 = input_doc.page_content.replace("\n\n", placeholder)
    step2 = step1.replace("\n", "").replace(" ", "").replace(".", "").replace('\x00', "")
    result = step2.replace(placeholder, "\n\n")
    cleaned_text = re.sub('[\ue000-\uf8ff]', '', result)
    final_text = re.sub(r'\s+', "", cleaned_text)
    threshold = 10
    cleaned_content = re.sub(
        r'[A-Za-z0-9$%!@#^&*]{' + str(threshold) + ',}', '', final_text)
    input_doc.page_content = cleaned_content
    input_doc.metadata = input_doc.metadata
    return input_doc


def ingest():
    """
    数据导入函数。

    这个函数用于导入数据，包括加载数据、拆分文本、清理字符串、创建向量存储等。

    Returns:
        创建的向量存储对象。
    """
    PATH = "data/vector_src"
    print("Loading data...")

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    persist_directory = PATH + "/" +  "lzm-vectorstore/"
    folder_path = "data/docs/" + "lzm/"  # path to xx_src
    loader = PyPDFDirectoryLoader(folder_path)
    text_docs = DirectoryLoader(path=folder_path, glob="**/*.txt").load()
    raw_documents = loader.load()
    raw_documents.extend(text_docs)

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=50,
        length_function=tiktoken_len,
    )
    documents = text_splitter.split_documents(raw_documents)

    documents = list(map(lambda doc: clean_string(doc), documents))
    print(documents[0])
    print("Creating vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def get_documents(question, db):
    """
    获取相关文档的函数。

    这个函数用于根据给定的问题从数据库中检索相关的文档。

    Args:
        question: 给定的问题字符串。
        db: 数据库对象。

    Returns:
        相关的文档列表。
    """
    retriver = db.as_retriever(search_type="mmr", search_kwargs={'k': 3})
    docs = retriver.get_relevant_documents(query=question)
    return docs
