#测试用的文件，不是正式的程序


import os
#当前文件路径
print(os.path.realpath(__file__))
#当前文件所在的目录，即父路径
print(os.path.split(os.path.realpath(__file__))[0])

#声明一个列表存储文件名
list_name =[]

path=os.path.split(os.path.realpath(__file__))[0]
for fileName in os.listdir(path):
      if os.path.splitext(fileName)[1] == '.csv':
            list_name.append(fileName)
list_name.sort()
print(list_name)



import pandas as pd #加载模块

# Pycharm控制台输出结果“部分内容省略”的解决方法
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.max_rows', 1000)



#读取csv

i=0;
for fileName in list_name:
      i=i+1
      data = pd.read_csv(fileName, encoding = 'gb18030',usecols=[0,8])
      data=data.rename(columns={"日期": "date", "涨跌额": os.path.splitext(fileName)[0]})
      if i==1:
            data_total=data;
      else:
            data_total=pd.merge(data_total,data,on='date')
# print(data_total)
data_total=data_total.sort_values(by='date')
# print(data_total)
data_total.reset_index(drop=True, inplace=True)
# print(data_total)


data_total=data_total.T
data_total=data_total.reset_index()
data_total=data_total.rename(columns={"index":"stockName"})
# print(data_total)
print(data_total.index)

column_list=[column for column in data_total]
column_list.remove('stockName')
# print('column_list',column_list)


data_total=data_total.drop(index=0)
data_total.reset_index(drop=True, inplace=True)
print(data_total.columns)
print(data_total.sort_values(by=0,axis=1))
#
# data_total=data_total.replace('None',0)
# data_total.to_csv("test.csv",index=False,sep=',')
#
# date_list=[column for column in data_total]
# date_list.remove('stockName')
# print(date_list)
# print(data_total.dtypes)
# print(data_total.sort(["2016-01-04"]))


# print(data_total)
# data_total.to_csv("test.csv",index=False,sep=',')


# date_list=[column for column in data_total]
# date_list.remove('date')
# print(date_list)
# print(data_total.sort_values(by=date_list[1]).loc[:,['date',date_list[1]]])


# # print(data.iloc[2,1])
# print(data.sort_values(by='日期'))
#
# data=data.sort_values(by='日期')
# # print(data)