import pandas as pd
import numpy as np

s = pd.Series([1,3,6,np.nan,4,1])
print(s)

dates = pd.date_range('20171216',periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4),index = dates,columns=['a','b','c','d'])
print(df)

print('df.dtypes is:\n',df.dtypes)
print('df.index is:\n',df.index)
print('df.columns is :\n',df.columns)

print(df.values)  # values of df
print(df.describe())  # count mean std min max and so on...

print(df.sort_index(axis=0,ascending=False))  #sort index

print(df.sort_values(by = 'b'))    # sort by 'b' columns

print(df['a'],df.a)  #效果相同
print(df[0:3],df['20171216':'20171219'])

# select data 

# select by label :loc
print(df.loc['20171216'].values)
print(df.loc[:,['a','b']])  #选出 'a' 'b' 两列的数据（所有行）

#select by position :iloc
print(df.iloc[3])

#mixed select: ix 
print(df.ix[:3,['a','b']])

#Boolean indexing
print(df[df.a>0.3])

#赋值
df.iloc[2,2] = 1111
df.loc['20171216','c'] = np.nan
df.a[df.a>0.1] = 666
df.b[df.a>0.1] = 888  #很灵活
#加一列
df['e'] = 0  #相同的
df['f'] = pd.Series([1,2,3,4,5,6],index = pd.date_range('20171216',periods =6)) #加一个序列，注意对齐

print(df)


# 处理丢失数据，例上面的NaN
# 丢掉
print(df.dropna(axis=0,how='any')) #how = ‘any’,‘all’  any 出现一个就丢，all 全出现异常才丢
# 把异常（丢失）数据置为某个值
print(df.fillna(value=555))

print(df.isnull())  #检查是否有Nan
print(np.any(df.isnull() == True)) #当数据很多时，检查是否有Nan

# 导入导出































