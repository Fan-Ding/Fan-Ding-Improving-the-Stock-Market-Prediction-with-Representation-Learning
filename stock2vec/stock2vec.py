import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#（一）网络结构
class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super(SkipGram,self).__init__()
        #embbeding层
        self.embeddings = nn.Embedding(n_vocab, n_embed)

        #输出层（线性层+softmax）
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    #前向传播
    def forward(self, inputs):
        embed = self.embeddings(inputs)
        #embeds = self.embeddings(inputs).view((1, -1))
        scores = self.output(embed)
        log_ps = self.log_softmax(scores)

        return log_ps

#（二）batch的准备，为unsupervised，准备数据获取（center,contex)的pair：

#get_batches函数会调用get_target函数
def get_target(words, idx, window_size=5):
    '''Get a list of words in a window around an index. '''
    #words：单词列表；idx：input word的索引号

    target_window = np.random.randint(1, window_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    target_words = set(words[start_point:idx] + words[idx + 1:end_point + 1])
    return list(target_words)
# 我们定义了一个get_targets函数，接收一个单词索引号，基于这个索引号去查找单词表中对应的上下文（默认window_size=5）。
# 请注意这里有一个小trick，我在实际选择input word上下文时，使用的窗口大小是一个介于[1, window_size]区间的随机数。
# 这里的目的是让模型更多地去关注离input word更近词。


def get_batches(words, batch_size, window_size=5):
    ''' 构建一个获取batch的生成器, Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words) // batch_size
    # // 表示整除

    # only full batches仅取full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_target(batch, i, window_size)

            #由于一个inputword会对应多个outputword
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y
        #yield函数详解见：https://blog.csdn.net/mieleizhi0522/article/details/82142856


#（二）训练过程:使用nn.NLLLoss()
#导入训练集
with open("sentences.txt")as file_object:
    sentences_list=file_object.readlines()
file_object.close()
print(len(sentences_list))

with open("stockName.txt")as file_object:
    stockNames=file_object.readline()
file_object.close()

vocab = stockNames.strip().split()
#set()函数创建一个无须不重复的函数集
print("vocab",vocab)


#需要定义,词汇到index的映射表
#制作word_to_ix字典
# vocab_to_int={}
vocab_to_int = {word: i for i, word in enumerate(vocab)}
print('vocab_to_int',vocab_to_int)


# 制作训练用的index表
# vocab_int是一个二维的list，每一行是一个sentence对应的 词汇index
vocab_int_list=[]
for sentence in sentences_list:
    vocab_int_list.append([vocab_to_int[w] for w in sentence.strip().split()])
print('vocab_int_list',vocab_int_list)


#初始化并指定网络的参数
# check if GPU is available，如果Gpu不可用就自动使用cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# you can change, if you want(embedding层的结点数)
embedding_dim = 25

model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
loss_function = nn.NLLLoss()
losses = []
optimizer = optim.Adam(model.parameters(), lr=0.001)

# print_every = 500
steps = 0
epochs = 200


# train for some number of epochs
for e in range(epochs):
    #total_loss的声明必须放在for循环内，因为关于list的append()函数，传是的相应值的地址
    total_loss = torch.Tensor([0])
    # get input and target batches
    for vocab_int in vocab_int_list:
        for inputs, targets in get_batches(vocab_int, 10):
            steps += 1
            inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            log_ps = model(inputs)
            #计算误差
            loss = loss_function(log_ps, targets)
            #清空之前步骤保留的梯度，把梯度设为零
            optimizer.zero_grad()
            #反向传播给每一个节点附上新计算的梯度
            loss.backward()
            #更新梯度
            optimizer.step()
            total_loss += loss.data
            # print(loss.data)
    print(total_loss)
    losses.append(total_loss)
print(losses)  # 在训练集中每次迭代损失都会减小!
print((model.embeddings.weight).detach().numpy())
# array=(model.embeddings.weight).detach().numpy()
array=model.embeddings.weight

similar_list=[]

count=0;
for row1 in array:
    for row2 in array:
        similarity = torch.cosine_similarity(row1, row2, dim=0)
        # print(similarity)
        if similarity< -0.2:
            count=count+1
        if similarity>0.2:
            count=count+1
        similar_list.append(similarity.detach().numpy())
similar_array=np.array(similar_list)
similar_array=similar_array.reshape(50,50)

# print(similar_list)
# print(similar_array)
print(count)

np.savetxt("word2vec_stock_similarity.txt",similar_array)
temp=np.loadtxt("word2vec_stock_similarity.txt")
# print(temp)


# with open("word2vec_stock_similarity.txt", 'w') as file_object:
#     for row in similar_array:
#         file_object.write(row + "\n")
# file_object.close()
# print(array.shape)