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


## Starts the SAS python

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import pandas as pd
d = pd.DataFrame([['0071', 'Patton' , 17, 27],
 ['1001', 'Joyner' , 13, 22], ['0091', 'Williams', 111, 121],
 ['0110', 'Jurat' , 51, 55]],
 columns = ['ID', 'Name', 'Before', 'After'])


# # Group by

# In[3]:


df = pd.DataFrame(
[['I', 'North', 'Patton', 17, 27, 22],
['I', 'South', 'Joyner', 13, 22, 19],
['I', 'East', 'Williams', 111, 121, 29],
['I', 'West', 'Jurat', 51, 55, 22],
['II', 'North', 'Aden', 71, 70, 17],
['II', 'South', 'Tanner', 113, 122, 32],
['II', 'East', 'Jenkindfs', 99, 99, 24],
['II', 'West', 'Milner', 15, 65, 22],
['III', 'North', 'Chang', 69, 101, 21],
['III', 'South', 'Gupta', 11, 22, 21],
['III', 'East', 'Haskins', 45, 41, 19],
['III', 'West', 'LeMay', 35, 69, 20],
['III', 'West', 'LeMay', 35, 69, 20]],
columns=['District', 'Sector', 'Name', 'Before', 'After', 'Age'])


# In[4]:


df.head(0)


# # Group by the nums columns

# In[33]:


def stats(group):
    return {'count': group.count(),
             'median' : group.median()}
bins=[10,50,100,113]
grp_labels=['11-50','50-100','above 100']
grp_labels


# In[34]:


df['b_format']=pd.cut(df['Before'], bins, labels=grp_labels)


# In[35]:


df.head(0)


# In[36]:


df['Before'].groupby(df['b_format']).apply(stats).unstack()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


gb=df.groupby(['District']).first()


# In[17]:


print(gb)


# In[7]:


d_grby_sum=df.groupby(['District']).sum()
d_grby_sum


# In[24]:


d_grby_sum.index


# In[25]:


d_grby_sum.columns


# In[26]:


gb


# In[27]:


gb.groups.keys()


# In[28]:


gb.groups


# # Iterations Over Groups

# In[29]:


gb=df.groupby(['District'])
gb


# In[39]:


gb=df.groupby(['District'])
for i,k in gb:
    print(i,k)


# In[43]:


gb=df.groupby(['District']).last()
gb


# # First. & Last.

# In[46]:


gb=df.groupby(['District']).last()
gb


# # Group Stats

# In[54]:


gb=df.groupby('District')
gb.agg({'Before':'mean','After':['mean','median','max']},)


# In[62]:


def ff(x):
    return x['Before'].std() > 5
df.groupby(['District']).filter(ff)

df['Before'].mean()


# In[64]:


def nam(x):
    return x['Before'].mean() < 50
df.groupby(['District']).filter(nam)


# In[66]:


def nam(x):
    return x['Before'].mean() > 50
df.groupby(['Before']).filter(nam)


# # Group by Columns with float numbers

# In[67]:


def stats(group):
    return {'count': group.count(),
             'median' : group.median()}
bins=[0,25,50,75,200]
gp_labels=['0 to 25', '26 to 50', '51 to 75', 'Over 75']


# In[71]:


df['Before_new']=pd.cut(df['Before'], bins, labels=gp_labels)


# In[72]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


d.reset_index('ID', inplace=True)


# In[14]:


d


# In[19]:


d.index


# In[20]:


print(d.index,'\n', d.columns)


# In[44]:


i = pd.DataFrame([['Patton' , 17, 27],
['Joyner' , 13, 22],
['Williams' , 111, 121],
['Jurat' , 51, 55],
['Aden' , 71, 70]])


# In[45]:


i


# In[36]:


i[0]


# In[46]:


df = pd.DataFrame([['I','North', 'Patton', 17, 27],
['I', 'South','Joyner', 13, 22],
['I', 'East', 'Williams', 111, 121],
['I', 'West', 'Jurat', 51, 55],
['II','North', 'Aden', 71, 70],
['II', 'South', 'Tanner', 113, 122],
['II', 'East', 'Jenkins', 99, 99],
['II', 'West', 'Milner', 15, 65],
['III','North', 'Chang', 69, 101],
['III', 'South','Gupta', 11, 22],
['III', 'East', 'Haskins', 45, 41],
['III', 'West', 'LeMay', 35, 69],
['III', 'West', 'LeMay', 35, 69]],
columns=['District', 'Sector', 'Name', 'Before',
'After'])


# In[41]:


df


# In[47]:


df[['District','Sector']].head(3)


# In[47]:


df[0:3:2]


# In[76]:


df.set_index('Name', drop=True, inplace=True)


# In[90]:


df


# In[78]:


print(df.index)


# In[86]:


df.loc['Patton':'Aden',:'Sector']


# In[91]:


df.loc['Haskins',['Sector']]


# #  Conditionals

# In[93]:


df.loc[(df['Sector']=='South')]


# In[97]:


df.loc[(df['Sector']=='South') & (df['After'] > 22)]


# In[104]:


df.reset_index(inplace=True)


# In[113]:


df.loc[df['Name'].str.endswith('r'),['Name','District','Before']]


# In[119]:


df.set_index('Name', inplace=True)


# In[120]:


df.loc[['Joyner','Gupta'], 'Before']


# In[121]:


df.loc[['Joyner','Gupta'], 'After']=1000


# In[122]:


df


# In[123]:


df.loc[:,'After']=10000000000000


# In[124]:


df


# # Return the index values based on the positional values

# In[128]:


df.iloc[0,]


# In[130]:


df.iloc[0]


# In[132]:


df.columns


# In[134]:


df.iloc[[2,10,12],:2]='Ashok Konatham'


# In[135]:


df


# In[143]:


df.iloc[:,:5]


# In[146]:


df.iloc[:-3,-1]


# In[147]:


import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.2f}'.format
cols = pd.MultiIndex.from_tuples([ (x,y) for x in
['Test1','Test2','Test3'] for y in ['Pre','Post']])

nl = '\n'
np.random.seed(98765)


# In[148]:


cols


# In[149]:


df = pd.DataFrame(np.random.randn(2,6),index = ['Row 1','Row 2'],
columns = cols)


# In[150]:


df


# In[8]:


d=pd.read_csv("https://raw.githubusercontent.com/RandyBetancourt/PythonForSASUsers/master/data/uk_accidents.csv", parse_dates=["Date"])


# In[12]:


d.head(5)


# In[15]:





# In[16]:


d['Time'].fillna(0)


# In[17]:


d.isnull().sum()


# In[5]:


d.info()


# In[9]:


d.tail(2)


# In[10]:


import matplotlib.pyplot as plt
d.hist(column="Age_of_Casualty", grid=False)
plt.show()


# In[7]:


d[["Date","Age_of_Casualty","Casualty_Severity"]].head(5)


# # create data frame

# In[12]:


import pandas as pd
df=pd.DataFrame([['ashok',2],['kumar', 2],['konatham',42]], columns=['Name','number'])


# In[18]:


df.loc[df.number==2,'number']=None
df


# In[27]:


y=df.fillna({'number':2})
u=y['number'].drop_duplicates()
u


# In[48]:


d.hist(column='Age_of_Casualty', grid=False)
pt.show


# In[71]:


df.loc[df.number==2, 'number']=None
df


# In[81]:


df.loc[df.number==None, 'number']=0


# In[82]:


df


# In[28]:


f=pd.DataFrame([['cold','slow', None, 2.7, 6.6, 3.1],
 ['warm', 'medium', 4.2, 5.1, 7.9, 9.1],
 ['hot', 'fast', 9.4, 11.0, None, 6.8],
 ['cool', None, None, None, 9.1, 8.9],
 ['cool', 'medium', 6.1, 4.3, 12.2, 3.7],
 [None, 'slow', None, 2.9, 3.3, 1.7],
 [None, 'slow', None, 2.9, 3.3, 1.7]],columns=[ 'Temp', 'Speed', 'Measure1', 'Measure2','Measure3', 'Measure4'])
f.dtypes


# In[112]:


for c in f.columns:
    print(c, sum(f[c].isnull()))


# In[114]:


for i in f.columns:
    print(i, sum(f[i].notnull()))


# In[116]:


f.isnull().sum()


# In[122]:


f['total']=f['Measure1']+f['Measure2']


# In[124]:


f['total'].isnull().sum()


# In[128]:


ff=f.dropna()


# In[129]:


ff=f.dropna(axis=1)


# In[132]:


ff=f.dropna(axis=1)
ff


# In[138]:


rr=f.drop_duplicates()
rr


# In[143]:


yy=f.dropna(thresh=0).isnull().sum()
yy


# In[150]:


fil=f.fillna(0)
f


# In[154]:


fil.loc[fil.Measure2==0,'Measure2']=10000


# In[164]:


ddd=f.fillna({'Temp':'slow', 'Speed':'love'})
ddd


# In[169]:


tt=f[['Measure1','Measure2']].fillna(f.Measure4.sum())
tt


# In[41]:


f.isnull().sum()


# In[118]:


f.isnull().sum()


# # Indexing

# In[52]:


g=pd.DataFrame([['0071', 'Patton' , 17, 27],
['1001', 'Joyner' , 13, 22],
['0091', 'Williams', 111, 121],
['0110', 'Jurat' , 51, 55]],
columns = ['ID', 'Name', 'Before', 'After'])
g


# In[53]:


g.set_index('ID', inplace=True, )


# In[54]:


g


# In[19]:


p=pd.DataFrame([['Patton' , 17, 27],['Joyner' , 13, 22],
 ['Williams' , 111, 121],['Jurat' , 51, 55],
 ['Aden' , 71, 70]])


# In[20]:


print(p.index, '\n', p.columns)


# In[63]:


p[0]


# In[65]:


df = pd.DataFrame([['I','North', 'Patton', 17, 27],
 ['I', 'South','Joyner', 13, 22],
 ['I', 'East', 'Williams', 111, 121],
 ['I', 'West', 'Jurat', 51, 55],
 ['II','North', 'Aden', 71, 70],
 ['II', 'South', 'Tanner', 113, 122],
 ['II', 'East', 'Jenkins', 99, 99],
 ['II', 'West', 'Milner', 15, 65],
 ['III','North', 'Chang', 69, 101],
 ['III', 'South','Gupta', 11, 22],
 ['III', 'East', 'Haskins', 45, 41],
 ['III', 'West', 'LeMay', 35, 69],
 ['III', 'West', 'LeMay', 35, 69]],
 columns=['District', 'Sector', 'Name', 'Before',
'After'])
df


# In[67]:


df[['Name','Before']].head()
df[['Name','Before']].isnull().sum()


# In[76]:


df[0:5:3]


# In[95]:


df.loc[0:3:2,'Name':'Before']


# In[2]:


import pandas as pd


# In[3]:


file = pd.read_csv("C:/Python/uk_accidents.csv")


# In[4]:


file.head(5)


# In[ ]:


from sqlalchemy import create_engine
engine = create_engine(f'teradata://{username}:{password}@tdprod:22/')


# In[ ]:


file1.to_sql('SQLTableFromDF',engine,if_exists='replace', chunksize=100, index=False)
#file1 is the dataframe name
#RDBSM target table to write, in this case its SQLTableFromDF
#Engine object containing the RDBMS connection string


# In[1]:


import pandas as pd


# In[2]:


file_loc="C:\\Python\\uk_accidents.csv"


# In[25]:


df=pd.read_csv(file_loc)


# In[6]:


print(df.shape)


# In[26]:


print(df.info())
df[['Date']].head(3)


# In[10]:


df['Casualty_Severity'].dtype


# In[20]:


df=pd.read_csv(file_loc,parse_dates=['Date'])


# In[18]:


print(df.info())


# In[22]:


df.tail(0)


# In[24]:


#slicing the dataframe
df[['Sex_of_Casualty','Date']].head(2)


# In[27]:


#histogram


# In[30]:


import matplotlib.pyplot as plt
df.hist(column='Speed_limit', grid=False)
plt.show()


# In[4]:


list=[1,2,3,4,]
for i in list:
    i=i+i
    print(i)


# In[6]:


import pandas as pd
import numpy as np


# In[30]:


df=pd.read_csv("https://raw.githubusercontent.com/RandyBetancourt/PythonForSASUsers/master/data/uk_accidents.csv",parse_dates=['Date'])


# In[34]:


print(df.info())


# In[ ]:





# In[15]:


print(df.shape)
print(df.info())


# In[12]:


print(df.describe())


# In[36]:


df.tail(5)


# In[37]:


print(df['Date'].tail(5))


# In[43]:


print(df[['Accident_Severity']].describe())


# In[49]:


import matplotlib.pyplot as plt
df.hist(column='Accident_Severity',grid=False)
plt.show()


# In[64]:


df=pd.DataFrame([['none',10],['Ashok',12]], columns=('chars','nums'))


# In[65]:


print(df.dtypes)


# In[66]:


df


# In[67]:


df.loc[df.nums==10,'nums'] =100000


# In[68]:


df


# In[3]:


import pandas as pd
df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},'B': {0: 1, 1: 3, 2: 5},'C': {0: 2, 1: 4, 2: 6}})


# In[9]:


df


# In[11]:


fd=pd.melt(df, id_vars=['A'])


# In[12]:


pd.melt()


# # big heading
# ## smal
# ##### adfjdkfjdlkfj

# In[7]:


s="hello"
t="world"
print(s+" "+t)


# In[9]:


print(s+" "+t.upper())


# In[12]:


print(t.count('o'))


# In[13]:


s[0]


# In[14]:


t[3]


# In[15]:


s[:5]


# In[16]:


s[1:5]


# In[17]:


s[1:]


# In[18]:


s="ashok konatham"
print(len(s))


# In[19]:


print(len("ashok kumar"))


# In[22]:


s[2:-5]


# In[27]:


q="ashok's kumara"
p='ashok\'s kumar'


# In[28]:


p


# In[1]:


import pandas as pd


# In[2]:


url_l="https://raw.githubusercontent.com/RandyBetancourt/PythonForSASUsers/master/data/Left.csv"


# In[1]:


import pandas as pd


# In[2]:


url_l="https://raw.githubusercontent.com/RandyBetancourt/PythonForSASUsers/master/data/Left.csv"


# In[3]:


l=pd.read_csv(url_l)
l=l.rename(columns={'ID':'PID'})
l.dtypes


# In[4]:


url_r="https://raw.githubusercontent.com/RandyBetancourt/PythonForSASUsers/master/data/Right.csv"


# In[5]:


r=pd.read_csv(url_r)


# In[6]:


r.dtypes


# In[7]:


pd.merge(l,r,left_on=['PID'],right_on=['ID'],how='outer',sort=False)


# In[ ]:





# In[9]:


left=pd.merge(l,r,how='left',right_on='ID',left_on='PID',sort=False)
left


# In[14]:


right=pd.merge(l,r,how='inner',right_on='ID',left_on='PID',sort=False)
right


# In[ ]:


outer=pd.merge(l,r,on='ID',how='outer',sort=False)


# In[ ]:


outer


# In[ ]:


right_only=pd.merge(l,r,on='ID',how='outer',sort=False,indicator='in_col')
right_only


# In[ ]:


right_only[right_only['in_col']=='right_only']


# In[ ]:


right_only[right_only['in_col']=='left_only']


# In[ ]:


oute_unmatched=right_only[(right_only['in_col']=='left_only') | (right_only['in_col']=='right_only') ]


# In[ ]:


oute_unmatched


# In[ ]:


outer_um=right_only[(right_only['in_col']!='both')]


# In[ ]:


outer_um


# In[ ]:


left=pd.merge(l,r,on='ID',how='outer',sort=False, indicator='in_col')


# In[ ]:


o=left[(left['in_col']=='right_only')]
o


# In[ ]:


gb=l.groupby(['ID']).count()
gb


# In[ ]:


notmat=pd.merge(l,r,on='ID',how='outer',sort=False,indicator='in_col')


# In[ ]:


notmat[(notmat['in_col']!='both')]


# In[ ]:


gb=notmat.groupby(['ID','Salary']).sum()


# In[ ]:


gb


# In[ ]:


l


# In[ ]:


l


# In[ ]:


left=pd.merge(l,r,on='ID',how='outer',sort=False, indicator='in_col')


# In[ ]:


left


# In[ ]:


left = pd.DataFrame(
 { 'Style' : ['S1', 'S2', 'S3', 'S4'],
 'Size' : ['SM', 'MD', 'LG', 'XL']},
index = ['01', '02', '03', '05'])
right = pd.DataFrame(
 { 'Color' : ['Red', 'Blue', 'Cyan', 'Pink'],
 'Brand' : ['X', 'Y', 'Z', 'J']},
 index = ['01', '02', '03', '04'])


# In[ ]:


j=left.join(right,how='inner')
j


# In[ ]:


jm=pd.merge(left, right, how='outer', right_index=True, left_index=True)


# In[ ]:


jm


# In[ ]:


left = pd.DataFrame(
{'Style' : ['S1', 'S2', 'S3', 'S4'],
'Size' : ['SM', 'MD', 'LG', 'XL'],
'Key' : ['01', '02', '03', '05']})
right = pd.DataFrame(
{'Color' : ['Red', 'Blue', 'Cyan', 'Pink'],
'Brand' : ['X', 'Y', 'Z', 'J']},
index = ['01', '02', '03', '04'])


# In[ ]:


j=left.join(right,on='Key',how='left')


# In[ ]:





# # Update

# In[15]:


import numpy as np
master=pd.DataFrame({'ID': ['023', '088', '099', '111'],
                     'Salary' : [45650, 55350, 55100, 61625]})
trans = pd.DataFrame({'ID': ['023', '088', '099', '111', '121'],
                      'Salary': [45650, 61000, 59100, 61625, 50000],
                      'Bonus': [2000, np.NaN , np.NaN, 3000, np.NaN]})


# In[16]:


master.update(trans,join='left')
master


# In[17]:


u=pd.merge(master,trans,on='ID',how='outer', suffixes=('_o','_n'))


# In[23]:


u.drop('Salary_o',axis=1)
u.rename(columns={'Salary_n':'NewVar'}).drop('Salary_o',axis=1)


# In[ ]:


df1=u.copy()


# In[ ]:


def tax(row):
    if row['Salary_n'] >= 50000:
        val=0
    else:
        val=1
    return val


# In[ ]:


df1['Taxs']=df1.apply(tax, axis=1)


# In[ ]:


df1


# In[ ]:


df1.loc[df1['Salary_n'] >= 50000,'Taxes']=1
df1.loc[df1['Salary_n'] < 50000, 'Taxes']=0


# In[ ]:


df1


# In[ ]:


df1.loc[df1['Salary_n']==50000,'T']=1


# In[ ]:


df1


# In[ ]:


df1=df1.drop(['Taxes','T'],axis=1)


# In[ ]:


df1


# In[ ]:


def tax(row):
    if row['Salary_n'] >= 50000:
        val=1
    else:
        val=0
    return val


# In[ ]:


df1['T']=df1.apply(tax, axis=1)


# In[ ]:


df1


# In[ ]:


df = pd.DataFrame({'ID': ['A0', 'A1', 'A2', 'A3', 'A4', '5A', '5B'],
'Age': [21, 79, 33, 81, np.NaN, 33, 33],
'Rank': [1, 2, 3, 3, 4, 5, 6]})
print(df)


# In[ ]:


df.sort_values(by=['Age','ID'])


# In[ ]:


df.sort_values(by=['Age'],na_position='first')


# In[ ]:


df.sort_values(by=['Age','ID'], na_position='first', ascending=(True,False))


# In[ ]:


loc1 = pd.DataFrame({'Onhand': [21, 79, 33, 81],
'Price': [17.99, 9.99, 21.00, .99]},
index = ['A0', 'A1', 'A2', 'A3'])

loc2 = pd.DataFrame({'Onhand': [12, 33, 233, 45],
'Price': [21.99, 18.00, .19, 23.99]},
index = ['A4', 'A5', 'A6', 'A7'])

loc3 = pd.DataFrame({'Onhand': [37, 50, 13, 88],
'Price': [9.99, 5.00, 22.19, 3.99]},
index = ['A8', 'A9', 'A10', 'A11'])
frames = [loc1, loc2, loc3]
all = pd.concat(frames)
print(all)


# In[ ]:


all = pd.concat(frames, keys=['Loc1', 'Loc2', 'Loc3'])


# In[ ]:


all.loc['Loc2']


# In[ ]:


all_parts = loc1.append([loc2, loc3])


# In[ ]:


all_parts


# In[ ]:


all_parts = pd.concat([loc1, loc2, loc3], join='outer')
all_parts


# In[ ]:


all_parts.sort_values(['Onhand','Price'],na_position='first', ascending=(False,True))


# In[ ]:


mask=all_parts.duplicated('Onhand')


# In[ ]:


dup=all_parts.loc[mask]


# In[ ]:


dup


# In[ ]:


mask=df.duplicated('Age',keep=False)


# In[ ]:


df.loc[mask]


# # Drop Duplicate

# In[ ]:


dd=df.drop_duplicates('Age',keep='first')


# In[ ]:


dd


# # Find duplicarte
# ## SAS nodup key = keep"first", noduprec= keep=False

# In[ ]:


mask=df.duplicated('Age',keep=False)


# In[ ]:


mask


# In[ ]:


dd=df.loc[mask]
dd


# In[ ]:


dd=df.drop_duplicates('Age',keep='first')


# In[ ]:


dd


# # Sampling

# In[ ]:


import numpy as np


# In[ ]:


np.random.seed(123)
df=pd.DataFrame({'value': np.random.randn(360)},
               index=pd.date_range('1976-09-12', freq='M', periods=360))


# In[ ]:


df.head()


# In[ ]:


sam1=df.sample(n=100,replace=False)


# In[ ]:


sam1.head()


# In[ ]:


sam2=df.sample(frac=.3, replace=True)
sam2.count()


# # convert data types

# In[ ]:


import pandas as pd
df=pd.DataFrame({'Str':['1','2','3','4','5'], 
                'int' : [1,2,3,4,5,]})
df.dtypes


# In[ ]:


df['Str']=df['Str'].astype(float)
df['int']=df['int'].astype(object)


# In[ ]:


df.dtypes


# In[ ]:


df


# # Map or Formats

# In[ ]:


df=pd.DataFrame({'num': [1,2,3,4,5,6,7]})


# In[ ]:


day={1:'Sun', 2:'mon', 3:'Tue', 4:'Web', 5:'Thu', 6:'Fri', 7:'Sat'}


# In[ ]:


df['new']=df['num'].map(day)
df


# # Transpose

# In[ ]:


uni = {'School' : ['NCSU', 'UNC', 'Duke'],
'Mascot' : ['Wolf', 'Ram', 'Devil'],
'Students' : [22751, 31981, 19610],
'City' : ['Raleigh', 'Chapel Hill', 'Durham']}


# In[ ]:


df_uni = pd.DataFrame(data=uni)
df_uni


# In[ ]:


dt=df_uni.T
dt


# # Rename

# In[ ]:


df.rename(columns={'num':'blo','new':'str'}, inplace=True)
df


# # pandas Readers and Writers

# In[24]:


import pandas as pd
url = "https://raw.githubusercontent.com/RandyBetancourt/PythonForSASUsers/master/data/messy_input.csv"
df1 = pd.read_csv(url, skiprows=2)


# In[25]:


df1


# In[27]:


miss={'Amount':[' ','NA']}


# In[28]:


df1=pd.read_csv(url,skiprows=2,na_values=miss)
df1


# In[32]:


df1.iloc[3:4]


# In[33]:


df1.dtypes


# In[42]:


df3 = pd.read_csv(url, skiprows=2, na_values={'Amount':[' ','NA']}, dtype={'ID' : object})


# In[49]:


df3


# In[59]:


import math

def strip_sign(x):
        y = x.strip()
        if not y:
            return math.nan
        else:
            if y[0] == '$':
                return float(y[1:])
            else:
                return float(y)


# In[68]:


df4 = pd.read_csv(url, skiprows=2, converters={'ID' : str, 'Amount':
strip_sign},parse_dates=['Date']).set_index('ID')
df4.dtypes


# ## column heading

# In[73]:


cols=['ID', 'Trans_Date', 'Amt', 'Quantity', 'Status', 'Name']
df7 = pd.read_csv(url, skiprows=3, na_values=miss,
converters={'ID' : str,'Amt':strip_sign},
parse_dates=['Trans_Date'], header=None, names=cols,
usecols=[0, 1, 2, 3, 4, 5]).set_index('ID')


# In[74]:


df7


# ## Read .xls Files

# In[88]:


df8 = pd.read_excel('/project/messy_inpu.xlsx', sheet_name='Trans1',skiprows=2,
                    converters={'ID' : str}, parse_dates={'Date' :['Month', 'Day' ,'Year']}, keep_date_col=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


l=pd.read_csv(url_l)
l=l.rename(columns={'ID':'PID'})
l.dtypes


# In[4]:


url_r="https://raw.githubusercontent.com/RandyBetancourt/PythonForSASUsers/master/data/Right.csv"


# In[5]:


r=pd.read_csv(url_r)


# In[6]:


r.dtypes


# In[7]:


pd.merge(l,r,left_on=['PID'],right_on=['ID'],how='outer',sort=False)


# In[ ]:





# In[9]:


left=pd.merge(l,r,how='left',right_on='ID',left_on='PID',sort=False)
left


# In[14]:


right=pd.merge(l,r,how='inner',right_on='ID',left_on='PID',sort=False)
right


# In[ ]:


outer=pd.merge(l,r,on='ID',how='outer',sort=False)


# In[ ]:


outer


# In[ ]:


right_only=pd.merge(l,r,on='ID',how='outer',sort=False,indicator='in_col')
right_only


# In[ ]:


right_only[right_only['in_col']=='right_only']


# In[ ]:


right_only[right_only['in_col']=='left_only']


# In[ ]:


oute_unmatched=right_only[(right_only['in_col']=='left_only') | (right_only['in_col']=='right_only') ]


# In[ ]:


oute_unmatched


# In[ ]:


outer_um=right_only[(right_only['in_col']!='both')]


# In[ ]:


outer_um


# In[ ]:


left=pd.merge(l,r,on='ID',how='outer',sort=False, indicator='in_col')


# In[ ]:


o=left[(left['in_col']=='right_only')]
o


# In[ ]:


gb=l.groupby(['ID']).count()
gb


# In[ ]:


notmat=pd.merge(l,r,on='ID',how='outer',sort=False,indicator='in_col')


# In[ ]:


notmat[(notmat['in_col']!='both')]


# In[ ]:


gb=notmat.groupby(['ID','Salary']).sum()


# In[ ]:


gb


# In[ ]:


l


# In[ ]:


l


# In[ ]:


left=pd.merge(l,r,on='ID',how='outer',sort=False, indicator='in_col')


# In[ ]:


left


# In[ ]:


left = pd.DataFrame(
 { 'Style' : ['S1', 'S2', 'S3', 'S4'],
 'Size' : ['SM', 'MD', 'LG', 'XL']},
index = ['01', '02', '03', '05'])
right = pd.DataFrame(
 { 'Color' : ['Red', 'Blue', 'Cyan', 'Pink'],
 'Brand' : ['X', 'Y', 'Z', 'J']},
 index = ['01', '02', '03', '04'])


# In[ ]:


j=left.join(right,how='inner')
j


# In[ ]:


jm=pd.merge(left, right, how='outer', right_index=True, left_index=True)


# In[ ]:


jm


# In[ ]:


left = pd.DataFrame(
{'Style' : ['S1', 'S2', 'S3', 'S4'],
'Size' : ['SM', 'MD', 'LG', 'XL'],
'Key' : ['01', '02', '03', '05']})
right = pd.DataFrame(
{'Color' : ['Red', 'Blue', 'Cyan', 'Pink'],
'Brand' : ['X', 'Y', 'Z', 'J']},
index = ['01', '02', '03', '04'])


# In[ ]:


j=left.join(right,on='Key',how='left')


# In[ ]:





# # Update

# In[15]:


import numpy as np
master=pd.DataFrame({'ID': ['023', '088', '099', '111'],
                     'Salary' : [45650, 55350, 55100, 61625]})
trans = pd.DataFrame({'ID': ['023', '088', '099', '111', '121'],
                      'Salary': [45650, 61000, 59100, 61625, 50000],
                      'Bonus': [2000, np.NaN , np.NaN, 3000, np.NaN]})


# In[16]:


master.update(trans,join='left')
master


# In[17]:


u=pd.merge(master,trans,on='ID',how='outer', suffixes=('_o','_n'))


# In[23]:


u.drop('Salary_o',axis=1)
u.rename(columns={'Salary_n':'NewVar'}).drop('Salary_o',axis=1)


# In[ ]:


df1=u.copy()


# In[ ]:


def tax(row):
    if row['Salary_n'] >= 50000:
        val=0
    else:
        val=1
    return val


# In[ ]:


df1['Taxs']=df1.apply(tax, axis=1)


# In[ ]:


df1


# In[ ]:


df1.loc[df1['Salary_n'] >= 50000,'Taxes']=1
df1.loc[df1['Salary_n'] < 50000, 'Taxes']=0


# In[ ]:


df1


# In[ ]:


df1.loc[df1['Salary_n']==50000,'T']=1


# In[ ]:


df1


# In[ ]:


df1=df1.drop(['Taxes','T'],axis=1)


# In[ ]:


df1


# In[ ]:


def tax(row):
    if row['Salary_n'] >= 50000:
        val=1
    else:
        val=0
    return val


# In[ ]:


df1['T']=df1.apply(tax, axis=1)


# In[ ]:


df1


# In[ ]:


df = pd.DataFrame({'ID': ['A0', 'A1', 'A2', 'A3', 'A4', '5A', '5B'],
'Age': [21, 79, 33, 81, np.NaN, 33, 33],
'Rank': [1, 2, 3, 3, 4, 5, 6]})
print(df)


# In[ ]:


df.sort_values(by=['Age','ID'])


# In[ ]:


df.sort_values(by=['Age'],na_position='first')


# In[ ]:


df.sort_values(by=['Age','ID'], na_position='first', ascending=(True,False))


# In[ ]:


loc1 = pd.DataFrame({'Onhand': [21, 79, 33, 81],
'Price': [17.99, 9.99, 21.00, .99]},
index = ['A0', 'A1', 'A2', 'A3'])

loc2 = pd.DataFrame({'Onhand': [12, 33, 233, 45],
'Price': [21.99, 18.00, .19, 23.99]},
index = ['A4', 'A5', 'A6', 'A7'])

loc3 = pd.DataFrame({'Onhand': [37, 50, 13, 88],
'Price': [9.99, 5.00, 22.19, 3.99]},
index = ['A8', 'A9', 'A10', 'A11'])
frames = [loc1, loc2, loc3]
all = pd.concat(frames)
print(all)


# In[ ]:


all = pd.concat(frames, keys=['Loc1', 'Loc2', 'Loc3'])


# In[ ]:


all.loc['Loc2']


# In[ ]:


all_parts = loc1.append([loc2, loc3])


# In[ ]:


all_parts


# In[ ]:


all_parts = pd.concat([loc1, loc2, loc3], join='outer')
all_parts


# In[ ]:


all_parts.sort_values(['Onhand','Price'],na_position='first', ascending=(False,True))


# In[ ]:


mask=all_parts.duplicated('Onhand')


# In[ ]:


dup=all_parts.loc[mask]


# In[ ]:


dup


# In[ ]:


mask=df.duplicated('Age',keep=False)


# In[ ]:


df.loc[mask]


# # Drop Duplicate

# In[ ]:


dd=df.drop_duplicates('Age',keep='first')


# In[ ]:


dd


# # Find duplicarte
# ## SAS nodup key = keep"first", noduprec= keep=False

# In[ ]:


mask=df.duplicated('Age',keep=False)


# In[ ]:


mask


# In[ ]:


dd=df.loc[mask]
dd


# In[ ]:


dd=df.drop_duplicates('Age',keep='first')


# In[ ]:


dd


# # Sampling

# In[ ]:


import numpy as np


# In[ ]:


np.random.seed(123)
df=pd.DataFrame({'value': np.random.randn(360)},
               index=pd.date_range('1976-09-12', freq='M', periods=360))


# In[ ]:


df.head()


# In[ ]:


sam1=df.sample(n=100,replace=False)


# In[ ]:


sam1.head()


# In[ ]:


sam2=df.sample(frac=.3, replace=True)
sam2.count()


# # convert data types

# In[ ]:


import pandas as pd
df=pd.DataFrame({'Str':['1','2','3','4','5'], 
                'int' : [1,2,3,4,5,]})
df.dtypes


# In[ ]:


df['Str']=df['Str'].astype(float)
df['int']=df['int'].astype(object)


# In[ ]:


df.dtypes


# In[ ]:


df


# # Map or Formats

# In[ ]:


df=pd.DataFrame({'num': [1,2,3,4,5,6,7]})


# In[ ]:


day={1:'Sun', 2:'mon', 3:'Tue', 4:'Web', 5:'Thu', 6:'Fri', 7:'Sat'}


# In[ ]:


df['new']=df['num'].map(day)
df


# # Transpose

# In[ ]:


uni = {'School' : ['NCSU', 'UNC', 'Duke'],
'Mascot' : ['Wolf', 'Ram', 'Devil'],
'Students' : [22751, 31981, 19610],
'City' : ['Raleigh', 'Chapel Hill', 'Durham']}


# In[ ]:


df_uni = pd.DataFrame(data=uni)
df_uni


# In[ ]:


dt=df_uni.T
dt


# # Rename

# In[ ]:


df.rename(columns={'num':'blo','new':'str'}, inplace=True)
df


# # pandas Readers and Writers

# In[24]:


import pandas as pd
url = "https://raw.githubusercontent.com/RandyBetancourt/PythonForSASUsers/master/data/messy_input.csv"
df1 = pd.read_csv(url, skiprows=2)


# In[25]:


df1


# In[27]:


miss={'Amount':[' ','NA']}


# In[28]:


df1=pd.read_csv(url,skiprows=2,na_values=miss)
df1


# In[32]:


df1.iloc[3:4]


# In[33]:


df1.dtypes


# In[42]:


df3 = pd.read_csv(url, skiprows=2, na_values={'Amount':[' ','NA']}, dtype={'ID' : object})


# In[49]:


df3


# In[59]:


import math

def strip_sign(x):
        y = x.strip()
        if not y:
            return math.nan
        else:
            if y[0] == '$':
                return float(y[1:])
            else:
                return float(y)


# In[68]:


df4 = pd.read_csv(url, skiprows=2, converters={'ID' : str, 'Amount':
strip_sign},parse_dates=['Date']).set_index('ID')
df4.dtypes


# ## column heading

# In[73]:


cols=['ID', 'Trans_Date', 'Amt', 'Quantity', 'Status', 'Name']
df7 = pd.read_csv(url, skiprows=3, na_values=miss,
converters={'ID' : str,'Amt':strip_sign},
parse_dates=['Trans_Date'], header=None, names=cols,
usecols=[0, 1, 2, 3, 4, 5]).set_index('ID')


# In[74]:


df7


# ## Read .xls Files

# In[88]:


df8 = pd.read_excel('/project/messy_inpu.xlsx', sheet_name='Trans1',skiprows=2,
                    converters={'ID' : str}, parse_dates={'Date' :['Month', 'Day' ,'Year']}, keep_date_col=True)


