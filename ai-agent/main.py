import os
from typing import List, Optional
from pydantic import BaseModel, Field
import math
from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import BaseChatModel
from langchain_community.chat_models import ChatDeepseek, ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from dotenv import load_dotenv

# 1. 计算器工具
class CalculatorInput(BaseModel):
    """计算器工具的输入模型，用于参数验证"""
    expression: str = Field(description="需要计算的数学表达式，例如'2 + 3 * 4'或'sqrt(16)'")

class CalculatorTool(BaseTool):
    """一个简单的计算器工具，支持基本的数学运算和函数"""
    name = "Calculator"
    description = """
    用于执行数学计算。当需要进行数值计算、解方程或使用数学函数时使用。
    可以处理加减乘除（+、-、*、/）、幂运算（**）和括号。
    支持的数学函数包括：sqrt(开方)、sin(正弦)、cos(余弦)、tan(正切)、log(对数)等。
    输入应为有效的数学表达式，例如：'2 + 3 * 4'、'sqrt(25)'、'(5 + 3) / 2'。
    """
    args_schema = CalculatorInput  # 使用定义的输入模型进行参数验证

    def _run(self, expression: str) -> str:
        """执行计算并返回结果"""
        try:
            # 限制可执行的函数，只允许math模块中的函数和基本运算
            allowed_functions = {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'exp': math.exp,
                'pow': math.pow
            }

            # 使用eval执行计算，限制命名空间以提高安全性
            result = eval(expression, {"__builtins__": None}, allowed_functions)
            return f"计算结果: {expression} = {result}"
        except SyntaxError:
            return f"语法错误：无效的数学表达式 '{expression}'"
        except NameError as e:
            return f"错误：不支持的函数或变量 '{str(e)}'"
        except Exception as e:
            return f"计算错误：{str(e)}"

    async def _arun(self, expression: str) -> str:
        """异步执行计算（与同步版本相同）"""
        return self._run(expression)

# 2. Agent配置和记忆系统
class AgentConfig(BaseModel):
    """Agent配置类，管理所有必要的参数"""
    llm_provider: str = Field(default="deepseek", description="LLM提供商，支持deepseek或openai")
    deepseek_api_key: str = Field(default=os.getenv("DEEPSEEK_API_KEY", ""), description="Deepseek API密钥")
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""), description="OpenAI API密钥")
    memory_type: str = Field(default="buffer", description="记忆类型: buffer 或 summary")
    temperature: float = Field(default=0.7, description="LLM温度参数，0-1之间，值越高输出越随机")

class MemoryManager:
    """记忆管理器，负责初始化和管理对话记忆"""
    @staticmethod
    def init_memory(llm, memory_type: str = "buffer"):
        """初始化记忆系统"""
        memory_kwargs = {
            "memory_key": "chat_history",  # 记忆在提示词中的键名
            "return_messages": True        # 返回消息对象而非字符串
        }

        if memory_type == "summary":
            # 摘要记忆：会自动总结对话历史，适合长对话
            return ConversationSummaryMemory(
                llm=llm,** memory_kwargs
            )
        else:
            # 缓冲记忆：直接存储完整对话历史，适合短对话
            return ConversationBufferMemory(**memory_kwargs
                                            )

# 3. AI Agent主类
class AIAgent:
    """具备记忆功能和计算器工具调用能力的AI Agent"""

    def __init__(self, config: Optional[AgentConfig] = None):
        # 初始化配置
        self.config = config or AgentConfig()

        # 初始化语言模型
        self.llm = self._init_llm()

        # 初始化记忆
        self.memory = MemoryManager.init_memory(self.llm, self.config.memory_type)

        # 初始化工具
        self.tools = self._init_tools()

        # 初始化Agent
        self.agent = self._init_agent()

    def _init_llm(self) -> BaseChatModel:
        """初始化语言模型，支持切换不同提供商"""
        if self.config.llm_provider == "deepseek":
            if not self.config.deepseek_api_key:
                raise ValueError("请设置Deepseek API密钥")
            return ChatDeepseek(
                api_key=self.config.deepseek_api_key,
                model_name="deepseek-chat",
                temperature=self.config.temperature
            )
        elif self.config.llm_provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("请设置OpenAI API密钥")
            return ChatOpenAI(
                api_key=self.config.openai_api_key,
                model_name="gpt-3.5-turbo",
                temperature=self.config.temperature
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {self.config.llm_provider}")

    def _init_tools(self) -> List[Tool]:
        """初始化工具列表"""
        # 创建计算器工具实例
        calculator_tool = CalculatorTool()

        # 将工具转换为Agent可使用的格式
        return [
            Tool(
                name=calculator_tool.name,
                func=calculator_tool.run,
                description=calculator_tool.description,
                args_schema=calculator_tool.args_schema
            )
        ]

    def _init_agent(self):
        """初始化Agent实例"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,  # 显示思考过程，便于调试
            handle_parsing_errors=True  # 处理解析错误
        )

    def chat(self, input_text: str) -> str:
        """与Agent对话"""
        try:
            response = self.agent.run(input_text)
            return response
        except Exception as e:
            return f"对话过程中出错: {str(e)}"

    def switch_llm(self, provider: str):
        """切换LLM提供商"""
        self.config.llm_provider = provider
        self.llm = self._init_llm()
        # 重新初始化依赖LLM的组件
        self.memory = MemoryManager.init_memory(self.llm, self.config.memory_type)
        self.agent = self._init_agent()
        return f"已切换LLM提供商为: {provider}"

# 4. 使用示例
def main():
    # 加载环境变量（如果有）
    load_dotenv()

    # 初始化配置
    config = AgentConfig(
        # 可以在这里直接设置API密钥，或通过环境变量获取
        # deepseek_api_key="your_deepseek_api_key",
        # openai_api_key="your_openai_api_key",
        memory_type="buffer"  # 使用缓冲记忆
    )

    # 创建Agent实例
    try:
        agent = AIAgent(config)
    except ValueError as e:
        print(f"初始化Agent失败: {e}")
        print("请确保已设置正确的API密钥")
        return

    # 交互循环
    print("AI计算器Agent已启动！")
    print("可用命令:")
    print("- 直接输入问题或计算需求")
    print("- 输入 'switch [provider]' 切换模型 (例如 'switch openai')")
    print("- 输入 'exit' 退出程序")

    while True:
        try:
            user_input = input("\n你: ")

            if user_input.lower() == "exit":
                print("再见！")
                break

            if user_input.lower().startswith("switch "):
                provider = user_input.split(" ")[1].strip().lower()
                print(agent.switch_llm(provider))
                continue

            # 获取Agent响应
            response = agent.chat(user_input)
            print(f"Agent: {response}")
        except KeyboardInterrupt:
            print("\n程序已中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
    