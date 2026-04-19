import asyncio
import sys
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend, FilesystemBackend
# from langchain_core.messages import (
#     AIMessage, AIMessageChunk,
#     ToolMessage, ToolMessageChunk,
#     HumanMessage, HumanMessageChunk,
#     SystemMessage, SystemMessageChunk,
#     FunctionMessage, FunctionMessageChunk,
#     ChatMessage, ChatMessageChunk,
#     RemoveMessage
# )
from langchain_core.messages import (
    AIMessageChunk
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from pydantic import BaseModel, SecretStr

from agent_mod.prompt import main_system_prompt

DB_URI = "postgresql://root:xirui528437@192.168.3.156:5432/xrai?sslmode=disable"


class MyContextSchema(BaseModel):
    user_id: str


# @click.command()
async def main():
    async with (
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
    ):
        await checkpointer.setup()
        await store.setup()

        current_script_dir = Path(__file__).resolve().parent
        project_root = current_script_dir.parent.parent
        # skills_dir = project_root.parent / "skills"
        skills_dir = current_script_dir.parent / "skills"

        workspace = project_root.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        print(f"🔍 项目根目录: {project_root}")
        print(f"🔍 Skills 目录: {skills_dir}")
        print(f"🔍 backend.root_dir: {workspace}")
        sub_dirs = [d for d in skills_dir.iterdir() if d.is_dir()]
        print(f"🔍 发现 {len(sub_dirs)} 个技能文件夹: {[d.name for d in sub_dirs]}")
        skill_count = 0
        for sub_dir in sub_dirs:
            skill_md = sub_dir / "SKILL.md"
            if skill_md.exists():
                size = skill_md.stat().st_size
                print(f"✅ 加载成功: {sub_dir.name}/SKILL.md ({size} 字节)")
                skill_count += 1
            else:
                print(f"❌ {sub_dir.name} 缺少 SKILL.md")
        print(f"📊 共加载 {skill_count} 个有效技能\n")
        agent = create_deep_agent(
            name="西瑞大聪明",
            model=ChatOpenAI(
                base_url="http://192.168.3.156:8000/v1",
                api_key=SecretStr("EMPTY"),
                model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
                streaming=True,
                temperature=0,  # 建议对话时稍微增加一点随机性
                max_retries=3,
            ),
            system_prompt=main_system_prompt,
            debug=True,
            # response_format=CustomResponseFormat,
            backend=lambda rt: CompositeBackend(
                # default=StateBackend(rt),
                default=FilesystemBackend(
                    root_dir=str(workspace),
                    virtual_mode=True,
                ),  # Short-term: ephemeral scratch space
                routes={
                    "/memory/": StoreBackend(
                        # rt,
                        namespace=lambda ctx: (ctx.context.user_id,),
                    )  # Long-term: persists across conversations
                }
            ),
            # backend=(lambda rt: StoreBackend(rt)),
            interrupt_on={
                "write_file": True,
                "edit_file": True,
                "read_file": False,
            },
            checkpointer=checkpointer,
            store=store,
            # subagents=[get_db_subagent(), get_mechanic_subagent()],
            skills=["skills"],
            memory=["/memories/AGENTS.md"],
            context_schema=MyContextSchema,
            # tools=get_common_tools() + get_sql_tools() + get_file_tools() + get_rag_tools()
        )
        print(agent.get_graph().draw_mermaid())
        while True:
            user_i = input("用户：")
            try:
                async for item in agent.astream(
                        input={"messages": [{"role": "user", "content": user_i}]},
                        config={"configurable": {"thread_id": str(uuid.uuid4()), "recursion_limit": 5}},
                        context=MyContextSchema(user_id="1"),
                        stream_mode="messages",
                        # print_mode="updates",
                ):
                    for i in item:
                        if isinstance(i, AIMessageChunk):
                            print(i.content, end="")
                print("\n")
            except Exception as e:
                print(e)
                raise e



if __name__ == '__main__':
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
