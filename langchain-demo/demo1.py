import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化模型
model = ChatOpenAI(
    model="mimo-v2-flash", # 或者是小米 Mimo 提供的特定模型名称
    api_key="sk-c1i40pgexythcv1jealg3tlrn3y6615ctlrx8op530eqwcel",
    # 替换为你获得的 Mimo 接口地址
    base_url="https://api.xiaomimimo.com/v1" 
)

# 后续使用方式完全一样
response = model.invoke("你好，请问你是谁？")
print(response.content)