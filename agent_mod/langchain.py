"""
langsmith # 用于记录 AI 的每一轮对话历史、消耗的 Token 数以及响应时间

langchain-core # 所有的接口定义
langchain：基于 langchain-core 构建的完整应用开发框架
langchain-classic #已经被淘汰的旧版 API
langchain-community  #大部分第三方集成
langchain-text-splitters # 专门用于文档切分的模块
  BaseDocumentTransformer
     TextSplitter
        RecursiveCharacterTextSplitter
langchain-mcp-adapters


invoke：同步阻塞式调用，一次性返回完整结果。等待模型全部生成完毕，再把最终消息给你，通常是一个完整的 AIMessage
stream：异步流式迭代调用，逐块返回 token。返回迭代器 / 流，每次是消息片段（AIMessageChunk）

# stream_mode
values 从头到尾所有消息 + 工具调用历史
messages
updates 增量的消息数据
"""


"""
langchain-core

embeddings
 Embeddings
BaseRetriever
 VectorStoreRetriever
vectorstores
 VectorStore
documents
 Document
 BaseDocumentCompressor
  EmbeddingsFilter
  LLMChainExtractor
  CrossEncoderReranker
  DocumentCompressorPipeline


prompts
 BasePromptTemplate
  PromptTemplate
  FewShotPromptTemplate

output_parsers
 BaseOutputParser
  BaseTransformOutputParser
   BaseCumulativeTransformOutputParser
    JsonOutputParser

document_loaders
 BaseLoader
  UnstructuredBaseLoader
   UnstructuredFileLoader
    UnstructuredExcelLoader

callbacks
 Callbacks


tools
 tool
 BaseToolkit

messages
 BaseMessage  所有完整消息的父类
  BaseMessageChunk
   HumanMessageChunk
   AIMessageChunk
   ToolMessageChunk
   ChatMessageChunk
   SystemMessageChunk
   FunctionMessageChunk
  HumanMessage  完整用户消息
  AIMessage  完整智能体消息
  ToolMessage  完整工具结果消息
  ChatMessage  完整通用对话消息
  SystemMessage  完整系统指令消息
  FunctionMessage  完整函数调用结果消息
  RemoveMessage  对话历史删除指令，非普通消息
"""



"""
三方集成
langchain-community
PyPDFLoader：都是继承自document_loaders.BaseLoader
TextLoader
WebBaseLoader
UnstructuredBaseLoader

langchain-google-vertexai
langchain-google-genai
langchain-voyageai
langchain-anthropic
langchain-ollama
langchain-deepseek
langchain-openai
langchain-huggingface
langchain-ibm

langchain-redis
langchain-elasticsearch
langchain-mongodb
langchain-milvus
langchain-chroma
langchain-unstructured
langchain-neo4j
"""