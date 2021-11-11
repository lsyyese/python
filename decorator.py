# 1, 装饰器原理
# 一个装饰器是一个需要另一个函数作为参数的函数

def my_shiny_new_decorator(a_function_to_decorate):
    '''在装饰器内部动态定义一个函数：wrapper(原意：包装纸).
       这个函数将被包装在原始函数的四周
       因此就可以在原始函数之前和之后执行一些代码'''

    def the_wrapper_around_the_original_function():
        # 把想要在调用原始函数前运行的代码放这里
        print("Before the function runs")
        # 调用原始函数（需要带括号）
        a_function_to_decorate()
        # 把想要在调用原始函数后运行的代码放这里
        print("After the function runs")

    '''直到现在，"a_function_to_decorate"还没有执行过.
       我们把刚刚创建的 wrapper 函数返回.
       wrapper 函数包含了这个函数，还有一些需要提前后之后执行的代码，
       可以直接使用了（It's ready to use!）'''
    return the_wrapper_around_the_original_function


def a_stand_alone_function():
    print("I am a stand alone function, don't you dare modify me")


a_stand_alone_function()

'''现在，你可以装饰一下来修改它的行为.
   只要简单的把它传递给装饰器，后者能用任何你想要的代码动态的包装
   而且返回一个可以直接使用的新函数'''
a_stand_alone_function_decorated = my_shiny_new_decorator(a_stand_alone_function)
a_stand_alone_function_decorated()
# outputs:
# Before the function runs
# I am a stand alone function, don't you dare modify me
# After the function runs


# 2,装饰器实例
def bread(func):
    def wrapper():
        print(".....bread.....")
        func()
        print(".....bread.....")

    return wrapper


def ingredients(func):
    def wrapper():
        print(".....ingredients.....")
        func()
        print(".....ingredients.....")

    return wrapper


def sandwich(food="--sandwich--"):
    print(food)

# 不使用装饰器
sandwich()
# output:--sandwich--


# 使用装饰器
@bread
@ingredients
def sandwich(food="--sandwich--"):
    print(food)

sandwich()
# output
# .....bread.....
# .....ingredients.....
# --sandwich--
# .....ingredients.....
# .....bread.....


# 3,给装饰器函数传参
def a_decorator_passing_arguments(function_to_decorate):
    def a_wrapper_accepting_arguments(arg1, arg2):
        print("I got args! Look:", arg1, arg2)
        function_to_decorate(arg1, arg2)

    return a_wrapper_accepting_arguments

# 当你调用装饰器返回的函数时，你就在调用wrapper，而给wrapper的参数传递将会让它把参数传递给要装饰的函数
@a_decorator_passing_arguments
def print_full_name(first_name, last_name):
    print("My name is", first_name, last_name)


print_full_name("Peter", "Venkman")
# outputs:
# I got args! Look: Peter Venkman
# My name is Peter Venkman


# 4,含参数的装饰器
def pre_str(pre=''):
    # old decorator
    def decorator(F):
        def new_F(a, b):
            print(pre + " input", a, b)
            return F(a, b)

        return new_F

    return decorator

# get square sum
@pre_str('^_^')
def square_sum(a, b):
    return a ** 2 + b ** 2

# get square diff
@pre_str('T_T')
def square_diff(a, b):
    return a ** 2 - b ** 2

print(square_sum(3, 4))
print(square_diff(3, 4))
# outputs:
# ('^_^ input', 3, 4)
# 25
# ('T_T input', 3, 4)
# -7


# 5.装饰“类中的方法”
def method_friendly_decorator(method_to_decorate):
    def wrapper(self, lie):
        lie = lie - 3  # very friendly, decrease age even more :-)
        return method_to_decorate(self, lie)

    return wrapper

class Lucy(object):
    def __init__(self):
        self.age = 32

    @method_friendly_decorator
    def say_your_age(self, lie):
        print("I am %s, what did you think?" % (self.age + lie))

l = Lucy()
l.say_your_age(-3)
# outputs: I am 26, what did you think?


# 6.不定参数
def a_decorator_passing_arbitrary_arguments(function_to_decorate):
    def a_wrapper_accepting_arbitrary_arguments(*args, **kwargs):
        print("Do I have args?:")
        print(args)
        print(kwargs)
        function_to_decorate(*args, **kwargs)

    return a_wrapper_accepting_arbitrary_arguments


@a_decorator_passing_arbitrary_arguments
def function_with_no_argument():
    print("Python is cool, no argument here.")


function_with_no_argument()
# outputs
# Do I have args?:
# ()
# {}
# Python is cool, no argument here.

@a_decorator_passing_arbitrary_arguments
def function_with_arguments(a, b, c):
    print(a, b, c)


function_with_arguments(1, 2, 3)
# outputs
# Do I have args?:
# (1, 2, 3)
# {}
# 1 2 3

@a_decorator_passing_arbitrary_arguments
def function_with_named_arguments(a, b, c, platypus="Why not ?"):
    print("Do %s, %s and %s like platypus? %s" % (a, b, c, platypus))

function_with_named_arguments("Bill", "Linus", "Steve", platypus="Indeed!")
# outputs
# Do I have args ? :
# ('Bill', 'Linus', 'Steve')
# {'platypus': 'Indeed!'}
# Do Bill, Linus and Steve like platypus? Indeed!


class Mary(object):
    def __init__(self):
        self.age = 31
    @a_decorator_passing_arbitrary_arguments
    def say_your_age(self, lie=-3):  # You can now add a default value
        print("I am %s, what did you think ?" % (self.age + lie))

m = Mary()
m.say_your_age()
# outputs
# Do I have args?:
# (<__main__.Mary object at 0xb7d303ac>,)
# {}
# I am 28, what did you think


# 7.装饰类
def decorator(aClass):
    class newClass:
        def __init__(self, age):
            self.total_display = 0
            self.wrapped = aClass(age)
        def display(self):
            self.total_display += 1
            print("total display", self.total_display)
            self.wrapped.display()
    return newClass

@decorator
class Bird:
    def __init__(self, age):
        self.age = age

    def display(self):
        print("My age is", self.age)

eagleLord = Bird(5)
for i in range(3):
    eagleLord.display()


# 8.property 装饰器
'''property 装饰器用于类中的函数，使得我们可以像访问属性一样来获取一个函数的返回值
   但不能用原有的方法去访问了'''
class XiaoMing:
    first_name = '明'
    last_name = '小'
    @property
    def full_name(self):
        return self.last_name + self.first_name

xiaoming = XiaoMing()
print(xiaoming.full_name)


# 9.staticmethod 装饰器
'''staticmethod 装饰器同样是用于类中的方法，这表示这个方法将会是一个静态方法，
   意味着该方法可以直接被调用无需实例化，
   但同样意味着它没有 self 参数，也无法访问实例化后的对象。'''
class XiaoMing:
    @staticmethod
    def say_hello():
        print('同学你好')
XiaoMing.say_hello()

# 实例化调用也是同样的效果
xiaoming = XiaoMing()
xiaoming.say_hello()

# 10.classmethod 装饰器
'''classmethod 依旧是用于类中的方法，这表示这个方法将会是一个类方法，意味着该方法可以直接被调用无需实例化，
   但同样意味着它没有 self 参数，也无法访问实例化后的对象。相对于 staticmethod 的区别在于它会接收一个指向类本身的 cls 参数。'''
class XiaoMing:
    name = '小明'
    @classmethod
    def say_hello(cls):
        print('同学你好， 我是' + cls.name)
        print(cls)
XiaoMing.say_hello()

# 11.wraps 装饰器
'''一个函数不止有他的执行语句，还有着 __name__（函数名），__doc__ （说明文档）等属性，我们之前的例子会导致这些属性改变'''
def decorator(func):
    def wrapper(*args, **kwargs):
        """doc of wrapper"""
        print('123')
        return func(*args, **kwargs)
    return wrapper
@decorator
def say_hello():
    """doc of say hello"""
    print('同学你好')
print(say_hello.__name__)
print(say_hello.__doc__)


'''由于装饰器返回了 wrapper 函数替换掉了之前的 say_hello 函数，导致函数名，帮助文档变成了 wrapper 函数的了,
   解决这一问题的办法是通过 functools 模块下的 wraps 装饰器'''
from functools import wraps
def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """doc of wrapper"""
        print('123')
        return func(*args, **kwargs)
    return wrapper
@decorator
def say_hello():
    """doc of say hello"""
    print('同学你好')
print(say_hello.__name__)
print(say_hello.__doc__)