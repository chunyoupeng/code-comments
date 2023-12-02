from enum import Enum
class MyLanguage(Enum):
    PYTHON = "python"
    JAVA = "java"
    CPP = "cpp"
    JS = "js"

CODE_PATH = "/home/dell/Projects/code-comments/data/Simple-Java-Calculator"
CODE_TEMPLATE = """
<context>
程序员在写了代码之后忘记给代码注释. 但是一个代码又需要适量易读，且优美的注释。并且重点在于class中的注释
</context>
<instruct>
你现在是一个程序员助手. 你的任务是给程序员的代码进行优美易读，且数量合适的中文注释，但是避免注释过多，会影响阅读。让用户能够快速的理解相关的代码。
只输出注释后的代码。不输出其它任何多余的内容，不添加或删除任何代码，仅仅是给定一个代码，你给我加上优美易读的注释就行，好好思考这一点.
</instruct>
Q:
class StreamHandler(BaseCallbackHandler):
def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text
def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)
A:
```python
class StreamHandler(BaseCallbackHandler):
\"\"\"
StreamHandler 类用于处理从语言模型（LLM）接收的新标记并更新一个容器的显示内容。

这个类继承自 BaseCallbackHandler，是一个回调处理器，专门用于处理和响应来自语言模型的新生成的标记。

Attributes:
    container: 容器对象，用于在其中显示文本。
    text: 初始文本，用于在容器中开始显示。默认为空字符串。

Methods:
    on_llm_new_token(token: str, **kwargs): 当从语言模型接收到新标记时调用。将新标记附加到现有文本并更新容器的显示。
\"\"\"

def __init__(self, container, initial_text=""):
    \"\"\"
    初始化 StreamHandler 实例。

    Args:
        container: 容器对象，用于显示来自语言模型的文本。
        initial_text: 可选；初始文本字符串，用于在容器中开始显示。默认为空字符串。
    \"\"\"
    self.container = container
    self.text = initial_text

def on_llm_new_token(self, token: str, **kwargs) -> None:
    \"\"\"
    当从语言模型接收到新标记时被调用的方法。

    这个方法将接收到的标记附加到类实例的文本属性上，并更新容器中的显示内容。

    Args:
        token: 从语言模型接收到的新标记字符串。
        **kwargs: 可选参数，用于额外的功能或处理。
    \"\"\"
    self.text += token
    self.container.markdown(self.text)
    ```
Q:
"""
