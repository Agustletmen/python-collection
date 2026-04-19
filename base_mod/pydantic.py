"""
基于类型注解的运行时数据校验库
1. Pydantic V1（纯 Python 实现，兼容 Python 3.7+）
2. Pydantic V2（核心用 Rust 重构，性能提升 5-100 倍，推荐使用，兼容 V1 大部分语法）

BaseModel：继承BaseModel的类，会自动获得「传入数据校验 + 类型强制转换 + 结构化输出」的能力
"""