pip install uv
uv init

uv sync
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --index-url https://mirrors.ustc.edu.cn/pypi/web/simple

# 添加新依赖到 pyproject.toml
uv add fastapi  # 运行依赖
uv add --dev pytest  # 开发依赖

# 移除依赖
uv remove requests

# 生成锁定文件（类似 poetry.lock）
uv lock