import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()


word2vec_stock_similarity=np.loadtxt("word2vec_stock_similarity.txt")

for x in range(50):
    for y in range (x,50):
        if x < y:
            similarity=word2vec_stock_similarity[x,y]
            print(x,y)
            if similarity < 0:
                similarity= similarity * -1
            if similarity > 0.2:
                G.add_edge(x,y,weight=similarity)

nx.write_weighted_edgelist(G,"stock.edgelist")

for x in range(50):
    G_each_stock=nx.Graph()
    for y in range(50):
        if(x!=y):
            similarity = word2vec_stock_similarity[x, y]
            print(x, y)
            if similarity < 0:
                similarity = similarity * -1
            if similarity > 0.2:
                G_each_stock.add_edge(x, y, weight=similarity)
    nx.write_weighted_edgelist(G_each_stock,'/Users/dingfan/Desktop/PyCharmProject/node2vec/graph/each_stock_graph/'
                                            +str(x)+'.edgelist')





elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.2]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=50)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,width=1)
# nx.draw_networkx_edges(G, pos, edgelist=esmall,width=6, alpha=0.5, edge_color='b', style='dashed')

# labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis('off')
plt.show()
