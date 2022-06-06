import argparse
import networkx as nx
import node2vecWalk

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd


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


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/stock.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/stock.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_true')
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        i = 0;
        for edge in G.edges():
            i = i + 1
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()
    # print("index",i)
    return G


def learn_embeddings():

    with open("node_sentences.txt")as file_object:
        node_sentences_list = file_object.readlines()
    file_object.close()
    print(len(node_sentences_list))

    #定义词汇表
    vocab=[]
    for i in range(50):
        vocab.append(str(i))
    print("vocab", vocab)

    # 需要定义,词汇到index的映射表
    # 制作word_to_ix字典
    # vocab_to_int={}
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    print('vocab_to_int', vocab_to_int)

    # 制作训练用的index表
    # vocab_int是一个二维的list，每一行是一个sentence对应的 词汇index
    vocab_int_list = []
    for sentence in node_sentences_list:
        vocab_int_list.append([vocab_to_int[w] for w in sentence.strip().split()])
    print(node_sentences_list)
    print('vocab_int_list', vocab_int_list)

    # 初始化并指定网络的参数

    # check if GPU is available，如果Gpu不可用就自动使用cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # you can change, if you want(embedding层的结点数)
    embedding_dim = 25

    model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
    loss_function = nn.NLLLoss()
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print_every = 500
    steps = 0
    epochs = 50

    # train for some number of epochs
    for e in range(epochs):
        # total_loss的声明必须放在for循环内，因为关于list的append()函数，传是的相应值的地址
        total_loss = torch.Tensor([0])
        # get input and target batches
        for vocab_int in vocab_int_list:
            for inputs, targets in get_batches(vocab_int, 10):
                steps += 1
                inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
                inputs, targets = inputs.to(device), targets.to(device)

                log_ps = model(inputs)

                # 计算误差
                loss = loss_function(log_ps, targets)

                # 清空之前步骤保留的梯度，把梯度设为零
                optimizer.zero_grad()

                # 反向传播给每一个节点附上新计算的梯度
                loss.backward()

                # 更新梯度
                optimizer.step()

                total_loss += loss.data
                # print(loss.data)
        print(total_loss)
        losses.append(total_loss)
    print(losses)  # 在训练集中每次迭代损失都会减小!
    print((model.embeddings.weight).detach().numpy())
    array = model.embeddings.weight

    similar_list = []

    count = 0;
    for row1 in array:
        for row2 in array:
            similarity = torch.cosine_similarity(row1, row2, dim=0)
            # print(similarity)
            similar_list.append(similarity.detach().numpy())
    similar_array = np.array(similar_list)
    similar_array = similar_array.reshape(50, 50)

    np.savetxt("node2vec_stock_similarity.txt", similar_array)

    np_frame=pd.DataFrame(similar_array)
    np_frame.to_csv('node2vec_stock_similarity.csv')

    embedding_array=(model.embeddings.weight).detach().numpy()
    np.savetxt("node2vec_embedding.txt",embedding_array)


    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = node2vecWalk.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    
    with open("node_sentences.txt", 'w') as file_object:
      for walk in walks:
        for node in walk:
            file_object.write(str(node) + " ")
        file_object.write("\n")
    file_object.close()
    learn_embeddings()



# learn_embeddings(walks)

if __name__ == "__main__":
    args = parse_args()
    main(args)
