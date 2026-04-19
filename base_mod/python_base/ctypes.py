"""
● ctypes 允许 Python 直接调用动态链接库（.dll, .so, .dylib）中的函数
● 支持大多数 C 数据类型，并能进行类型转换
● 无需编写额外的 C 扩展，直接在 Python 中操作

C 类型	ctypes 类型
int	c_int
float	c_float
double	c_double
char	c_char
char*	c_char_p（字节串）
void*	c_void_p
int*	POINTER(c_int)
void	None（返回值）
"""