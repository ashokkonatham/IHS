av=3
x=5

for i in range(x):
      if i >= av:
         print("no more candies")
         break
      print("candies")

print("bye")

for i in range(1,10):
    if i%3==0:
        continue
    print(i)


for i in range(1,10):
    if i%2==0:
        pass
    else:
        print(i)


from numpy import *
arr=array([1,2,3,4])
print(arr)

arr=arr+5
from numpy import *

arr=array([
         [1,2,3,4],
         [4,5,6,7]
         ])
print(arr.ndim)

mat=matrix('1,2,3,4;1,2,3,4')

print(diagonal(mat))


def suare(a):
    return a*a
result=suare(5)
print(result)

#ananymous functions are called lamda(which are used just once)
f= lambda a:a*a
print(f(7))

f= lambda a,b: a+b
print(f(5,6))

def is_even(n):
    return n%2==0

num=[1,2,3,4,5,6,7,8]
even=list(filter(is_even,num))
print(even)

# or

ev=list(filter(lambda n:n%2==0,num))
print(ev)


print(__name__)

class computer:

    def __init__(self,cpu,ram):
        self.cpu=cpu
        self.ram=ram
    def config(self):
        print("Ashok's CPU & RAM::", self.ram, self.cpu)

comp1=computer("I7","10gb")

comp1.config()


#heap memory objects are stored and every object has address
class computer:

    def __init__(self):
        self.name="ashok"
        self.age=42
    def update(self):
        self.age=30

c1=computer()
c2=computer()

c1.name="bujji"
c1.age=35

c1.update()

print(c1.name)
print(c2.name)
print(c1.age)


class computer:
    def __init__(self):
        self.name="Ashok"
        self.age=29
    def update(self):
        self.age=30
    def compare(self,other):
        if self.age==other.age:
            return True
        else:
            return False
c1=computer()
c1.age=40
c2=computer()

if c1.compare(c2):
    print("they are same")
else:
    print("they are different")

#different type of variables 1. class variable & 2. instance varaible or instance namespace

class cars:
    wheel=4
    def __init__(self):
        self.name="BMW"
        self.model=2021
c1=cars()
print(c1.name,c1.model,cars.wheel)

#type of methods
#3 types of methods, 1. Instance, 2. Class and 3. Static

class student:
    #class variable
    school="Nice"

    #constructor
    def __init__(self,m1,m2,m3):
        #instance variable
        self.m1=m1
        self.m2=m2
        self.m3=m3

    #instance method
    def avg(self):
        return (self.m1+self.m2+self.m3)/3
    # Instance method - accessor
    def getm(self):
        self.m1=m1
    # Instacne method - mutator(can set the value)
    def set(self,value):
        self.m1=value

    @classmethod
    def info(cls):
        return cls.school

    @staticmethod
    def inf():
        print("Hey I got static methos")



s1=student(98,87,100)
s2=student(100,100,100)
s1.set=92

print(s1.m1,s1.m2,s1.m3,student.school,s1.avg(),s2.avg())
print(s1.m1,s1.set)
print(student.info())
student.inf()

# init special varible use

class computer:
    def __init__(self,val,bul):
        self.cpu=val
        self.ram=bul
    def config(self):
        print("Hey got it :", self.cpu, self.ram)
comp1=computer('i7','10gb')

print(comp1.config())

#class
#objects
#Inheritance : class ingeritance, class A, class B(A), Class(A,B) ,Super(), MRO(read from left to right)
#polymorphism - 1. Deck Typing, 2.Operator Overloading, 3. Method Overloading , 4. Method Overwriting

# Duck typing  : calling the objects from different class with same methods


class PyCharm:
    def execute(self):
        print("Duck Typing")

class Spider:
    def execute(self):
        print("Anusha")

class now:
    def __init__(self,ide):
        self.execute()

lap1=PyCharm()

lap2=Spider()

lap1.execute()
lap2.execute()



# polymorphism : 1.Duck typing, 2. Overloading

a=5
b=6

print(a+b)
print(int.__add__(a,b))

#here add, sub ects methods are called when we use operators like +, - called magic methods

class student:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    def __add__(self, other):
        m1 = self.m1 + other.m2
        m2 = self.m2 + other.m1
        s3 = student(m1, m2)

        return s3

    def __gt__(self, other):
        r1 = self.m1 + self.m1
        r2 = other.m1 + other.m2
        if r1 > r2:
            return True
        else:
            return False

    def __str__(self):
        return '{} {} '.format(self.m1,self.m2)

s1 = student(100, 98)
s2 = student(56, 98)
s3 = s1 + s2

print(s3.m1)

if s1 > s2:
    print("s1 wins")
else:
    print("s2 wins")

a=9
print(a.__str__())

print(s1)
print(s2)

# above example is called polymorphism "operator overloading" __add__ etc are called Magic methods

# method overloading & method overriding, in python there is no method overloading, we workaround passing
# none values

class din:
    def __init__(self,m1,m2):
        self.m1=m1
        self.m2=m2

    def sum(self,a=None,b=None,c=None):
        s=0
        if a!=None and b!=None and c!=None:
            s=a+b+c
        elif a!=None and b!=None:
            s=a+b
        else:
            s=a
        return s

s1=din(10,20)

print(s1.sum(1,2))

## method overriding

class a:
    def nnt(self):
        print("zingo")

class b(a):
    def nnt(self):
        print("bingo")

p=b()
p.nnt()

# abstract class (ABC = abstract base class)

class computer:
    def process(self):
        pass
    # this method without any-body(of method) is called Abstract method
 # ABC is the class having one Abstrat method in a class, python does not support Abstract method
 # we need to import them by a module ABC with "abstract method"
 # Iterator, has function of iter(), fyi next()
# Genrator uses the value as YEILD replacing the RETURN, this helps in calling large values data one by one
# Exception Handling
a=5
b=0

try:
    print(a/b)
except Exception as e:
    print(e)
    # Finally statement will close the resources after the execution of the code
finally:
    print("close the resource or db connections")

#### Multithreading, for parallel processing we use this

from time import sleep # this will make the output to sleep, for one sec : sleep(1)
from threading import *

class hello(Thread):
    def run(self):
        for i in range(5):
            print("hello")
            sleep(2)

class hi(Thread):
    def run(self):
        for i in range(5):
            print("hi")
            sleep(2)
t1=hello()
t2=hi()

t1.start()
sleep(0.5)
t2.start()

t1.join()
t2.join()
#join() will ensure the complete the all classes to complete

print("bye")
# here run method is inbuilt and start calling will call the run method
# Implementation with lanuages underneet: cpythom, jython, ironpython (dot net),pypy












