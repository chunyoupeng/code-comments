
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# 导入所需的模块

# 创建一个聊天模型
chat_model = ChatOpenAI(openai_api_base="https://aiapi.xing-yun.cn/v1",
        openai_api_key="sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a",
        model_name="gpt-3.5-turbo")

# 使用聊天模型创建一个对话链
conversation = ConversationChain(llm=chat_model)

# 开始对话
response = conversation.run("Hi, how can I assist you?")

# 打印响应结果
print(response)
