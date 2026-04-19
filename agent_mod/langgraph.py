"""
deepagents
deepagents-cli

langgraph
langgraph-checkpoint
langgraph-prebuild
langgraph-sdk

State（状态）：全局共享数据容器（用 TypedDict 定义），存储记忆、计划、步骤、结果
Node（节点）：执行单元（函数），输入 State、输出更新片段
Edge（边）：流程控制：普通边（固定跳转）/ 条件边（动态路由）
Graph：工作流容器，StateGraph 构建、compile 编译
Thread：会话实例，用于持久化、历史回溯、多会话隔离。同一个 thread_id 会有多个 checkpoint（每次对话都会保存一个）
CheckPoint：StateSnapshot 对象
Store


StateGraph
CompiledStateGraph
Pregel：底层执行引擎（支持并行超步）


add_node(name, func)
add_edge(from_node, to_node)
add_conditional_edges


SubAgent
CompiledSubAgent
InterruptOnConfig
BaseChatModel
BaseTool
ToolRuntime
AgentMiddleware
BaseCache
BaseStore
BaseCheckpointSaver
ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
BackEnd
 BackendProtocol
 BackendFactory

StateBackend - 短暂的内存存储，文件存在于Agent的状态中（与对话一起检查点）
FilesystemBackend - 真实的文件系统访问，支持虚拟模式（沙盒化到根目录）
StoreBackend - 持久的跨对话存储，使用 LangGraph 的 BaseStore 实现持久性，按 assistant_id 命名空间
CompositeBackend - 将不同的路径路由到不同的后端
"""