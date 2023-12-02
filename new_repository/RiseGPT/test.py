
from utils import *

import os

# 设置环境变量 OPENAI_API_BASE 和 OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = "https://aiapi.xing-yun.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a"

# 调用 ingest() 函数
ingest()
