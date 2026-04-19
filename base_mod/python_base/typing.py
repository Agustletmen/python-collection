"""
typing 容器是早期为补充原生容器 “无法标注元素类型” 的临时方案，3.9+ 原生容器支持下标标注后
● Python 3.8 及以下：原生容器（如 list）不支持下标标注（list[int] 会报错），必须用 typing 容器（List[int]）才能标注元素类型；
● Python 3.9+：原生容器支持下标标注（list[int]），typing 容器仍保留（作为兼容层），两者均可使用。

类型	作用
List	列表（指定元素类型）
Tuple	元组（指定元素类型 / 长度）
Dict	字典（指定键值类型）
Set/FrozenSet	集合 / 不可变集合
Iterable	可迭代对象
Iterator	迭代器
Optional	可选类型（允许 None）
Union	联合类型（多类型可选）
Any	任意类型
Callable	可调用对象（函数 / 方法）
Generic	泛型基类
TypeVar	自定义类型变量
Literal	字面量类型
TypedDict	类型化字典
"""