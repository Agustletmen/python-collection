class Student:  # 类名一般首字母大写
    num = 0  # 类属性

    def __init__(self, name, school):  # 构造方法
        self.name = name  # 对象属性
        self.school = school
        Student.num += 1

    def __del__(self):  # 析构方法，调用del()会触发，但是不建议使用，可能导致内存泄漏或逻辑错误
        pass

    def introduce(self):  # 对象方法，公有
        print('我是', self.name, '来自', self.school)

    def _foo(self): # _私有
        pass
    def __foo(self): # 在类内部会被转换为_类名__foo的形式，用于避免子类中的命名冲突
        pass

    @classmethod
    def count(cls):  # 类方法
        print('学生个数：', cls.num)

    @staticmethod
    def sayHello():  # 静态方法
        print('Hello')
        pass


"""
在实际编程中，几乎不会用到类方法和静态方法，因为我们完全可以使用函数代替它们实现想要的功能，
但在一些特殊的场景中（例如工厂模式中），使用类方法和静态方法也是很不错的选择。
"""

s = Student("王二麻子", '花城二小')
Student.sayHello()  # 静态方法的调用，既可以使用类名，也可以使用类对象
s.introduce()  # 对象方法的调用
Student.count()  # 类方法推荐使用类名直接调用，当然也可以使用实例对象来调用（不推荐）
