import re

#简单python匹配

pattern1 = "cat"
pattern2 ="bird"

string = 'dog runs to cat'

print(pattern1 in string)  #True
print(pattern2 in string)  #False

#用正则表达式

print(re.search(pattern1,string))
print(re.search(pattern2,string))

#匹配多种可能
#
ptn = r'r[au]n'  #加了r就变成了正则表达式，其中[]表示ran,run都行
print(re.search(ptn,string)) 

#匹配更多种可能
#
print(re.search(r'r[A-Z]n','dog runs to cat')) 
print(re.search(r'r[a-z]n','dog runs to cat')) 
print(re.search(r'r[0-9]n','dog r2ns to cat')) 
print(re.search(r'r[0-9a-z]n','dog runs to cat'))

# \d 数字形式的匹配 即[0-9]
#
print(re.search(r'r\dn','run r4n'))

# \D 非数字形式，\d的反
# 
print(re.search(r'r\Dn','run r4n'))


#空白
#
#\s 所有的空白 [\t \n \r \f \v]
#\S \s的反
print(re.search(r'r\sn','r\nn r4n'))
print(re.search(r'r\Sn','r\nn r4n'))


# \w :[a-zA-Z0-9_]
# \W 除了上述内容都是
# 
print(re.search(r'r\wn',r'r3n r$n'))
print(re.search(r'r\Wn',r'r3n r$n'))


#空白字符
#\b 字符的前一个或后一个
#\B 字符的前后，前后只要有就行
#
print(re.search(r'\bruns\b',r'dog runs to cat'))
print(re.search(r'\B runs',r'dog  runs to cat'))

#特殊字符，任意字符
#
# \\: 匹配\
#  .:匹配出\n的任何东西
print(re.search(r'runs\\',r'dog runs\ to cat'))
print(re.search(r'r.n',r'r#n to me'))

#句尾句首
#^:匹配句首
#$：匹配句尾
#
print(re.search(r'^dog',r'dog runs to cat'))
print(re.search(r'dog$',r'dog runs to cat'))


#是否有
#?:
#
print(re.search(r'Mon(day)?',r'Mon'))
print(re.search(r'Mon(day)?',r'Monday'))

#多行匹配
#
string1 ="""
dog runs to cat.
I run to dog.
"""

print(re.search(r'^I',string1))
print(re.search(r'^I',string1,flags = re.M))   #多行匹配

# 出现0次或多次
# *
# 
print(re.search(r'ab*','a'))
print(re.search(r'ab*','abbbbbbbbbbbbbbbbb'))


# 出现1次或多次
# +
# 
print(re.search(r'ab+','a'))
print(re.search(r'ab+','abbbbbbbbbbbbbbbbb'))


#可选次数
#{n,m} n至m次
#
# 出现0次或多次
# *
# 
print(re.search(r'ab{2,10}','a'))
print(re.search(r'ab{2,10}','abbbbbbbbbbbbbbbbb'))


#group
#
match = re.search(r'(\d+), Date: (.+)','ID: 02153, Date: Feb/12/2017')   #(\d+) 数字出现一次或的多次 (.+)匹配出空格外所有字符
print(match.group())  #所有内容
print(match.group(1)) #第一个括号的内容
print(match.group(2)) 


match = re.search(r'(?P<id>\d+), Date: (?P<date>.+)','ID: 02153, Date: Feb/12/2017')   #给各个group加名字
print(match.group('id')) 
print(match.group('date'))   

#寻找所有匹配
#findall
#
print(re.findall(r'r[ua]n','run ran ren'))

# |:或
# 
print(re.findall(r'(rnu|ran)','run ran ren'))

#替换
#re.sub()
#
print(re.sub(r'r[au]ns','catches','dog runs to cat'))

#分裂
#re.split()
#
print(re.split(r'[,;\.]','a;b,c.d;e'))  #碰到[]中的任何内容就分裂


#compile
#
compiled_re = re.compile(r'r[ua]n')
print(compiled_re.search('dog runs to cat'))













