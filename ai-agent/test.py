# a = 1
# print(a.__class__)
#
# print(dir(object))

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# 1. 配置 API 密钥和基础URL
os.environ["OPENAI_API_KEY"] = "sk-818ab4f7bfa94a1298fb009c776c02f8"  # 这里填DeepSeek的API密钥
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"  # DeepSeek的API地址

# 2. 初始化模型（用ChatOpenAI兼容模式）
chat_model = ChatOpenAI(
    model_name="deepseek-chat",  # 模型名称
    temperature=0.7,
    max_tokens=1024
)

# 3. 创建提示模板
prompt = ChatPromptTemplate.from_template(
    "请解释 {topic} 的基本概念，并列举2个应用场景。"
)

# 4. 创建并运行链
chain = LLMChain(llm=chat_model, prompt=prompt)
result = chain.run(topic="机器学习")

print("模型回答：")
print(result)


