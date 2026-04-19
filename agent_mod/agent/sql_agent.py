from pydantic import SecretStr


DB_URI = "postgresql://root:xirui528437@192.168.3.156:5432/xrai?sslmode=disable"

if __name__ == '__main__':
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent
    from langchain_openai import ChatOpenAI

    # 1. 连接数据库
    db = SQLDatabase.from_uri(DB_URI)
    llm=ChatOpenAI(
        base_url="http://192.168.3.156:8000/v1",
        api_key=SecretStr("EMPTY"),
        model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        streaming=True,
        temperature=0,  # 建议对话时稍微增加一点随机性
        max_retries=3,
    )
    # 2. 创建 SQL Agent
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    # 3. 运行查询
    while True:
        user_i = input("用户：")
        try:
            response = agent_executor.invoke({"input": user_i})
            print(response)
        except Exception as e:
            log.error(e)
            raise e