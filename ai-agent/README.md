# AI应用开发
Microsoft AutoGen
LangChain（适合 LLM 驱动的 Agent）
AutoGPT（自主任务分解）
MetaGPT（多 Agent 协作）
CrewAI

LlamaIndex
Haystack

mcp：Model Context Protocol
fastmcp


短期记忆：LangChain 的ConversationBufferMemory（简单存储对话历史）。
长期记忆：LangChain 的ConversationSummaryMemory（用 LLM 自动摘要长对话）。
持久化：结合数据库（如 Redis、SQLite）存储记忆，避免重启丢失。


模型（Models）：封装了各种 LLM（如 GPT-3.5/4、Claude 等）或聊天模型。
提示（Prompts）：用于格式化输入给模型的模板，支持动态填充变量。
链（Chains）：将多个组件（如提示 + 模型）组合起来完成复杂任务。
记忆（Memory）：保存对话历史，让模型能进行多轮对话。
代理（Agents）：让模型自主决定调用哪些工具（如搜索引擎、计算器）来完成任务。
工具（Tools）：代理可以调用的外部能力（如 API、数据库查询等）。

langchain                0.3.27
langchain-community      0.3.27
langchain-core           0.3.72
langchain-text-splitters 0.3.9