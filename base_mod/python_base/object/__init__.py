"""
Python 万物皆对象，所有内置数据类型都是 object 的子类，自定义的 class 默认也继承 object


所有类型（包括内置类型、自定义类型）本质上都是类（class）的实例，
其变量，本质上都是对应类的实例（对象），拥有属性（attribute）和方法（method）。

自定义类型（class）
字节串（bytes/bytearray）
模块（module）
函数（function）
范围（range）
NoneType（None）
Set/frozenset
Dictionary
Tuple
List
string
int/float/complex
bool


所有对象都满足面向对象的基本特征：
1. 身份：通过 id() 函数获取的唯一标识（内存地址）。
2. 类型：决定对象支持的操作（如 int 支持加减，str 支持拼接）。
3. 值：对象存储的数据（如 10、"hello"）。
"""

"""
Python 中的不可变对象包括：
● int
● float
● bool
● tuple
● string
之所以叫这 5 种为不可变对象，就是因为一旦变量名和堆中的某个地址绑定以后，再想修改该变量的值，就会重新再将该变量名和另一个堆地址进行绑定。
"""