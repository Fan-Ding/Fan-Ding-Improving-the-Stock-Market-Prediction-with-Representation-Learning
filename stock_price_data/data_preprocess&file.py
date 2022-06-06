import pandas as pd
import numpy as np
#加载模块
# Pycharm控制台输出结果“部分内容省略”的解决方法
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.max_rows', 1000)


import os
#当前文件路径
print(os.path.realpath(__file__))
#当前文件所在的目录，即父路径
print(os.path.split(os.path.realpath(__file__))[0])
#声明一个列表存储文件名
fileList_name =[]

path=os.path.split(os.path.realpath(__file__))[0]
for fileName in os.listdir(path):
      if os.path.splitext(fileName)[1] == '.csv':
            fileList_name.append(fileName)
fileList_name.sort()
print(fileList_name)
#打印50只股票的名称

with open("stockName.txt", 'w') as file_object:
      for file in fileList_name:
            file_object.write(os.path.splitext(file)[0] + " ")
file_object.close()
#存储50支股票的名称到txt文件中



#读取csv
i=0;
for fileName in fileList_name:
      i=i+1
      data = pd.read_csv(fileName, encoding = 'gb18030',usecols=[0,8])
      data=data.rename(columns={"日期": "date", "涨跌额": os.path.splitext(fileName)[0]})
      if i==1:
            data_total=data;
      else:
            data_total=pd.merge(data_total,data,on='date')
# print(data_total)
data_total=data_total.sort_values(by='date')
print(data_total)
data_total.reset_index(drop=True, inplace=True)
print(data_total)


data_total=data_total.T
data_total=data_total.reset_index()
data_total=data_total.rename(columns={"index":"stockName"})
print(data_total)
# print(data_total.index.values)

column_list=[column for column in data_total]
column_list.remove('stockName')
# print('column_list',column_list)

for column in column_list:
      data_total=data_total.rename(columns={column: data_total.iloc[0,column+1]})
data_total=data_total.drop(index=0)
data_total.reset_index(drop=True, inplace=True)
print(data_total)

data_total=data_total.replace('None',np.nan)
# data_total.to_csv("test.csv",index=False,sep=',')

date_list=[column for column in data_total]
date_list.remove('stockName')
print(date_list)
# print(data_total.dtypes)

with open("sentences.txt", 'w') as file_object:
      for date in date_list:
            data_total[date] = pd.to_numeric( data_total[date] )

            data_total.sort_values(by=[date])
            temp_data = data_total.sort_values(by=[date])
            temp_data = temp_data.loc[:, ['stockName', date]]
            temp_data = temp_data.dropna(axis=0, how='any')
            print(temp_data)

            stockList = temp_data['stockName'].values.tolist()

            for stock in stockList:
                  file_object.write(stock + " ")
            file_object.write('\n')
file_object.close()


# print(data_total.sort_values(by=['2016-01-04']))
# temp_data=data_total.sort_values(by=['2016-01-04'])
# temp_data=temp_data.loc[:,['stockName','2016-01-04']]
# temp_data=temp_data.dropna(axis=0,how='any')
# print(temp_data['stockName'].values.tolist())
# stockList=temp_data['stockName'].values.tolist()

# fileName="sentences.txt"
# with open(fileName,'a') as file_object:
#       for stock in stockList:
#             file_object.write(stock+" ")


# print(data_total)
# data_total.to_csv("test.csv",index=False,sep=',')


# print(data.iloc[2,1])
# test="""600048 600276 600115 000776 600061 000166 000002 002415 600663 600111 600104 000001 002739 600016 600028 600000 600606 002673 600015 000538 600010 002304 000063 000895 600036 600585 600340 600023 001979 000333 000858 600030 600018 000069 002736 002252 600637 600019 000651 000725 600519 600518 600485 300059 002027 002594 000625 300104 600050 002024 """
# vocab=set(test.split())
# print(vocab)