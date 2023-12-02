
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from utils import *
import json
import os 
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

_template = """Answer the user questions based on the context. 
<context>
{context}
<context/>
"""
# Prompt


os.environ["OPENAI_API_BASE"] = "https://aiapi.xing-yun.cn/v1" 
os.environ["OPENAI_API_KEY"] = "sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a"
st.title("RiseGPT")
with st.expander("ℹ️ 说明"):
    st.caption(
        "重庆市西南大学Rise实验室刘志明老师论文助手(第一次加载请稍等)"
    )

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-1106-preview"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = ingest()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Maximum allowed messages
max_messages = (
    100  # Counting both user and assistant messages, so 10 iterations of conversation
)


class StreamHandler(BaseCallbackHandler):
    """
    StreamHandler 类用于处理从语言模型（LLM）接收的新标记并更新一个容器的显示内容。

    这个类继承自 BaseCallbackHandler，是一个回调处理器，专门用于处理和响应来自语言模型的新生成的标记。

    Attributes:
        container: 容器对象，用于在其中显示文本。
        text: 初始文本，用于在容器中开始显示。默认为空字符串。

    Methods:
        on_llm_new_token(token: str, **kwargs): 当从语言模型接收到新标记时调用。将新标记附加到现有文本并更新容器的显示。
    """

    def __init__(self, container, initial_text=""):
        """
        初始化 StreamHandler 实例。

        Args:
            container: 容器对象，用于显示来自语言模型的文本。
            initial_text: 可选；初始文本字符串，用于在容器中开始显示。默认为空字符串。
        """
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        当从语言模型接收到新标记时被调用的方法。

        这个方法将接收到的标记附加到类实例的文本属性上，并更新容器中的显示内容。

        Args:
            token: 从语言模型接收到的新标记字符串。
            **kwargs: 可选参数，用于额外的功能或处理。
        """
        self.text += token
        self.container.markdown(self.text)



class StreamHandler(BaseCallbackHandler):
    """
    StreamHandler 类用于处理从语言模型（LLM）接收的新标记并更新一个容器的显示内容。

    这个类继承自 BaseCallbackHandler，是一个回调处理器，专门用于处理和响应来自语言模型的新生成的标记。

    Attributes:
        container: 容器对象，用于在其中显示文本。
        text: 初始文本，用于在容器中开始显示。默认为空字符串。

    Methods:
        on_llm_new_token(token: str, **kwargs): 当从语言模型接收到新标记时调用。将新标记附加到现有文本并更新容器的显示。
    """

    def __init__(self, container, initial_text=""):
        """
        初始化 StreamHandler 实例。

        Args:
            container: 容器对象，用于显示来自语言模型的文本。
            initial_text: 可选；初始文本字符串，用于在容器中开始显示。默认为空字符串。
        """
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        当从语言模型接收到新标记时被调用的方法。

        这个方法将接收到的标记附加到类实例的文本属性上，并更新容器中的显示内容。

        Args:
            token: 从语言模型接收到的新标记字符串。
            **kwargs: 可选参数，用于额外的功能或处理。
        """
        self.text += token
        self.container.markdown(self.text)


def remain_last(input_str):
    """
    从给定的字符串中提取最后一个部分。

    这个函数将给定的字符串按照斜杠（/）进行分割，然后返回最后一个部分。

    Args:
        input_str: 给定的字符串。

    Returns:
        最后一个部分的字符串。
    """
    parts = input_str.split("/")
    desired_part = parts[-1]
    base_name, _ = os.path.splitext(desired_part)
    return base_name



def get_response(question, docs):
    """
    获取对给定问题的回答。

    Args:
        question: 问题字符串。
        docs: 文档列表。

    Returns:
        回答字符串。
    """

    # 将文档内容连接成一个字符串作为上下文
    context = "\n".join([doc.page_content for doc in docs])

    # 创建一个 StreamHandler 实例
    stream_handler = StreamHandler(st.empty()) 

    # 创建一个 ChatOpenAI 实例
    chat_model = ChatOpenAI(
            openai_api_base="https://aiapi.xing-yun.cn/v1",
            openai_api_key="sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a",
            streaming=True,  # ! important
            callbacks=[stream_handler], # ! important
            model_name=st.session_state.openai_model,
        )

    # 创建 ChatPromptTemplate 实例
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # 获取历史消息
    list_history = st.session_state.messages
    list_history_str = json.dumps(list_history, ensure_ascii=False)

    # 创建 ConversationBufferMemory 实例并保存上下文
    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context({"input":"hi"},{"output": list_history_str})

    # 加载内存变量
    print(memory.load_memory_variables({}))

    # 创建 RunnablePassthrough 实例
    chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | chat_model
    )

    # 调用模型获取回答
    response = chain.invoke({"input": question, "context": context})
    return response.content

# persist_directory = "data/vector_src/lzm-vectorstore"

# 检查消息数量是否超过限制
if len(st.session_state.messages) >= max_messages:
    st.info(
        "您的使用次数过多了，请休息一下，休息完后，请重新打开网页，继续使用"
    )
else:
    # 获取用户输入的问题
    if question := st.chat_input("请输入您对刘志明老师文章的疑问，希望能帮您解惑"):
        # 获取相关文档
        docs = get_documents(question, st.session_state.db)

        # 将用户输入的问题添加到消息列表中
        st.session_state.messages.append({"role": "user", "content": question})

        # 在聊天界面显示用户输入的问题
        with st.chat_message("user"):
            st.markdown(question)



with st.chat_message("assistant"):
    full_response = ""
    response = get_response(question, docs)
    full_response += response
    sources = "\n\n".join([f"📚 来源 {i + 1}: { remain_last( d.metadata['source'] ) } 第 {d.metadata['page']}页" for i, d in enumerate(docs)])
    message_placeholder = st.empty()
    message_placeholder.markdown(sources)
st.session_state.messages.append(
    {"role": "assistant", "content": full_response}
)
