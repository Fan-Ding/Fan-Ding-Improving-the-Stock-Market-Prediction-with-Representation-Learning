import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


df_similarity = pd.read_csv("node2vec_stock_similarity.csv", index_col=0)

dir_str_stock = '/Users/dingfan/Desktop/PyCharmProject/learning-model/final_stock'
file_name_stock = os.listdir(dir_str_stock)
file_name_stock.sort()
# print(file_name_stock)
file_dir_stock = [os.path.join(dir_str_stock, x) for x in file_name_stock]
# print(file_dir_stock)


def generate_df_affect_by_n_days(series, series_y, n, index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = pd.DataFrame()
    for i in range(n):
        df['c%d' % i] = series.tolist()[i:-(n - i)]
    df['y'] = series_y.tolist()[n:]
    if index:
        df.index = series.index[n:]
    return df


def get_correlation_stock_data(target_stock_name):

    for i in range (len(file_dir_stock)):
        if (target_stock_name==file_dir_stock[i].split('/')[-1].split('.')[0]):
            target_stock_id = i

    print('target_stock_id:', target_stock_id)
    target_stock_pd = pd.read_csv(file_dir_stock[target_stock_id], index_col=0)
    target_stock_pd['sentiment']=target_stock_pd['sentiment'].fillna(999)



    sentiment_index=target_stock_pd.columns.get_loc("sentiment")
    change_index=target_stock_pd.columns.get_loc("change")
    for row in range(len(target_stock_pd.index)):
        sentiment= target_stock_pd.iloc[row,sentiment_index]
        change = target_stock_pd.iloc[row, change_index]
        if(sentiment==999):
            if(change>0):
                target_stock_pd.iloc[row,sentiment_index] = 2
            else:
                target_stock_pd.iloc[row,sentiment_index] = 0


    # # target_stock_pd.to_csv('000001_final.csv')
    # # correlation_sentinment=target_stock_pd['sentiment']
    return target_stock_pd

def readData(target_stock_name, column='data', n=5, all_too=True, index=False, train_end=-300):

    # print(get_correlation_sentiment(target_stock_name))

    # df = pd.read_csv(target_stock_name+".csv", index_col=0)  # 直接将第一列作为索引，不额外添加列。
    # df.index = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), df.index))

    df=get_correlation_stock_data(target_stock_name)
    drop_subset = ['confidence', 'negative_prob', 'positive_prob','stock','closing price','highest price','lowest price','opening price','before closing','turnover volume','turnover','total market capitalization','circulation market value']

    # print(df['stock'].isnull().value_counts())
    # df['stock']=df['stock'].fillna('-9999')
    # index_list=df[df.stock=='-9999'].index.to_list()
    # print(index_list)
    # df.drop(index_list,inplace=True)

    # df['sentiment']=df['sentiment'].fillna(1)

    df.replace('None',np.nan,inplace=True)
    df.drop(axis=1, columns=drop_subset, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)



    change_index=df.columns.get_loc("change")
    for row in range(len(df.index)):
        change = df.iloc[row, change_index]
        if (change > 0):
            df.iloc[row,change_index] = 1
        else:
            df.iloc[row, change_index] = 0

    ###############
    # df.to_csv('000001-temp.csv')
    print(df.dtypes)
    #############

    for tap in list(df):
        temp_numpy = df[tap]

        temp_numpy_max = np.max(temp_numpy)
        temp_numpy_min = np.min(temp_numpy)
        temp_numpy = (temp_numpy - temp_numpy_min) / (temp_numpy_max - temp_numpy_min)
        df[tap] = pd.Series(temp_numpy)

    df['data'] = None
    for row in range(len(df.index)):
        list_temp = []
        for col in range(len(df.columns) - 1):
            list_temp.append(df.iloc[row, col])
        str_temp = '#'.join(str(x) for x in list_temp)
        df.iloc[row, -1] = str_temp


    df_column = df[column].copy()  # 读取所要的数据
    df_y = df['change'].copy()

    train_end= -len(df_column.index)//3

    df_column_train, df_column_test = df_column[:train_end], df_column[train_end - n:]                                  #拆分训练数据集和测试集
    # df_column_train, df_column_test = df_column, df_column[train_end - n:]
    df_generate_from_df_column_train = generate_df_affect_by_n_days(df_column_train, df_y[:train_end], n, index=index)  # 生成训练用的时间序列
    df_generate_from_df_column_test  = generate_df_affect_by_n_days(df_column_test,df_y[train_end - n:], n , index=index)

    # if all_too:
    #     return df_generate_from_df_column_train, df_column, df.index.tolist()
    return df_generate_from_df_column_train, df_generate_from_df_column_test

class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 2)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        return out


class TrainSet(Dataset):
    def __init__(self, data):
        # 定义好 image 的路径
        self.data, self.label = data[:, :-1].float(), data[:, -1].long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class TestSet(Dataset):
    def __init__(self, data):
        # 定义好 image 的路径
        self.data, self.label = data[:, :-1].float(), data[:, -1].long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)



def rnn_model(i):

    target_stock_name = file_dir_stock[i].split('/')[-1].split('.')[0]

    n = 4
    LR = 0.0005
    EPOCH = 150
    train_end = -200
    inputsize = 4

    # 数据集建立

    df_train, df_test = readData(target_stock_name,'data', n=n, train_end=train_end)

    df_train_numpy = np.array(df_train)
    temp_list = []
    rows_num = len(df_train_numpy)

    print(df_train_numpy)

    i = 0
    for x in range(len(df_train_numpy)):
        for y in range(len(df_train_numpy[0]) - 1):
            item = df_train_numpy[x][y]
            item = item.split('#')
            item = list(map(lambda x: float(x), item))
            temp_list = temp_list + item
            print(i)
            i = i + 1
        temp_list = temp_list + [df_train_numpy[x][y + 1]]

    df_train_numpy = np.array(temp_list)
    df_train_numpy = df_train_numpy.reshape((rows_num, -1))

    print(df_train_numpy)
    df_train_tensor = torch.from_numpy(df_train_numpy)
    trainset = TrainSet(df_train_tensor)

    print(trainset.__getitem__(0))
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

    #################################
    df_test_numpy = np.array(df_test)
    temp_list = []
    rows_num = len(df_test_numpy)

    print(df_test_numpy)

    i = 0
    for x in range(len(df_test_numpy)):
        for y in range(len(df_test_numpy[0]) - 1):
            item = df_test_numpy[x][y]
            item = item.split('#')
            item = list(map(lambda x: float(x), item))
            temp_list = temp_list + item
            print(i)
            i = i + 1
        temp_list = temp_list + [df_test_numpy[x][y + 1]]

    df_test_numpy = np.array(temp_list)
    df_test_numpy = df_test_numpy.reshape((rows_num, -1))

    print(df_test_numpy)
    df_train_tensor = torch.from_numpy(df_test_numpy)
    testset = TrainSet(df_train_tensor)

    print(testset.__getitem__(0), testset.__len__())

    # rnn = torch.load('rnn.pkl')

    rnn = RNN(inputsize)

    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    losses = []

    for step in range(EPOCH):
        # total_loss的声明必须放在for循环内，因为关于list的append()函数，传是的相应值的地址
        total_loss = torch.Tensor([0])
        for tx, ty in trainloader:
            tx = tx.view(-1, n, inputsize)
            output = rnn(tx)
            loss = loss_func(output, ty)
            total_loss += loss.data
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
        print(step, total_loss)
        losses.append(total_loss)
        # if step % 10:
        #     torch.save(rnn, '/Users/dingfan/Desktop/PyCharmProject/learning-model/trained-model_0/'+target_stock_name+'rnn.pkl')
    torch.save(rnn.state_dict(),
               '/Users/dingfan/Desktop/PyCharmProject/learning-model/trained_model_group2/' + target_stock_name + 'rnn.pkl')

    test_x = testset.data
    test_y = testset.label.numpy()

    test_output = rnn(test_x.view(-1, n, inputsize))
    pred_y = torch.max(test_output, 1)[1].numpy()
    print(pred_y)
    print(test_y)

    # count = 0.0
    # for i in range(testset.__len__()):
    #     if (pred_y[i] == test_y[i]):
    #         count = count + 1
    # accuracy = count / testset.__len__()
    # with open('trained-result-group1', 'a')  as file_object:
    #     file_object.write(target_stock_name + " accuracy: " + str(accuracy) + '\n')
    #
    # print('accuracy', accuracy, count)

    # TP    predict 和 label 同时为1

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    TP += ((test_y == 1) & (pred_y == 1)).sum()
    # TN    predict 和 label 同时为0
    TN += ((test_y == 0) & (pred_y == 0)).sum()
    # FN    predict 0 label 1
    FN += ((test_y == 1) & (pred_y == 0)).sum()
    # FP    predict 1 label 0
    FP += ((test_y == 0) & (pred_y == 1)).sum()

    print(TP, TN, FN, FP)

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

    with open('trained-result-group2', 'a')  as file_object:
        file_object.write(target_stock_name + " accuracy: " + str(acc) + " F1: " + str(F1) + " " + str(TP) + " " + str(
            TN) + " " + str(FN) + " " + str(FP) + " " + str(TP + TN + FP + FN) + '\n')
    print('accuracy: ', acc, ' F1: ', F1)


if __name__=='__main__':

    for i in range(0,50):
        rnn_model(i)








