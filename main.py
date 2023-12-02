import os
import shutil
from pprint import pprint 
import warnings
import re
warnings.filterwarnings("ignore")
from langchain.document_loaders import TextLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter
from langchain.text_splitter import Language
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from constants import *

def process_repository(source_repo, new_repo):
    # 遍历源仓库中的所有文件和目录
    for root, dirs, files in os.walk(source_repo):
        # 对每个文件进行处理
        for file in files:
            file_path = os.path.join(root, file)
            pprint(file_path)
            # 检查文件扩展名是否为代码文件
            if file_path.endswith(('.py', '.java', '.cpp', '.c')):
                # 读取并处理代码
                processed_code = process(root, file)
                # 构造新仓库中的相同路径
                new_file_path = file_path.replace(source_repo, new_repo)
                new_file_dir = os.path.dirname(new_file_path)
                pprint(f"new_file_dir is {new_file_dir}")
                if not os.path.exists(new_file_dir):
                    os.makedirs(new_file_dir)
                # 将处理后的代码写入新仓库
                with open(new_file_path, 'w') as new_file:
                    new_file.write(processed_code)
            else:
                # 对于非代码文件，直接复制
                new_file_path = file_path.replace(source_repo, new_repo)
                new_file_dir = os.path.dirname(new_file_path)
                if not os.path.exists(new_file_dir):
                    os.makedirs(new_file_dir)
                shutil.copy(file_path, new_file_path)


def process(root, file):
    # 这里是处理代码的函数，具体实现依据需要填写
    # 示例：简单地返回原始代码
    # loader = GenericLoader.from_filesystem(
    #     root,
    #     glob=file,
    #     suffixes=[".py", ".cpp", ".java", ".c", ".js"],
    #     parser=LanguageParser(),
    # )
    loader = TextLoader(os.path.join(root, file))
    documents = loader.load()
    print(documents)
    splitter = get_splitter(file)
    # splitter = TextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = splitter.split_documents(documents)
    # print(f"the current file is {file}")
    # print("\n".join([text.page_content for text in texts]))
    processed_code = "\n\n".join([get_comment(text.page_content) for text in texts])
    return processed_code

def get_splitter(file):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=0
    )

    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=2000, chunk_overlap=0
    )
    cpp_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CPP, chunk_size=2000, chunk_overlap=0
    )
    Javascript_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=2000, chunk_overlap=0
    )
    match file.split('.'):
        case [_, 'py']:
            return python_splitter
        case [_, 'java']:
            return java_splitter
        case [_, 'cpp']:
            return cpp_splitter
        case [_, 'js']:
            return Javascript_splitter
        case _:
            return None
        
def get_comment(code_str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CODE_TEMPLATE),
            ("human", "{input}")
        ]
    )
    llm = ChatOpenAI(
        openai_api_base="https://aiapi.xing-yun.cn/v1",
        openai_api_key="sk-RSAL5bknVmekLf005e714770B4Af431d821397F97d865cEb",
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    # llm = Ollama(model="yi:34b-chat")
    chain = prompt | llm
    response = chain.invoke({"input": code_str})
    print(f"Response is {response.content}")
    rt = extract_code(response.content)
    return rt


def extract_code(text):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return ""
    return "\n".join(matches) 

def main():
    # 设置源仓库和新仓库的路径
    source_repo_path = CODE_PATH
    new_repo_path = './data/new_repo'  

    # 确保新仓库目录存在
    if not os.path.exists(new_repo_path):
        os.makedirs(new_repo_path)

    # 处理仓库
    process_repository(source_repo_path, new_repo_path)

    pprint("仓库处理完成。原仓库文件已复制并处理到新仓库。")

if __name__=='__main__':
    main()