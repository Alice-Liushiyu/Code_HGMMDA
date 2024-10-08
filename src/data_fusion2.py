import torch
import numpy as np
import numpy as np
import random
import networkx as nx
from collections import defaultdict
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)
# # 读取语义相似度矩阵
# with open('D:\Mamba\data\ds.txt', 'r') as f:
#     semantic_similarity = np.array([list(map(float, line.strip().split())) for line in f])
#
# # 读取高斯核相似度矩阵
# with open('D:\Mamba\data\dg.txt', 'r') as f:
#     gaussian_similarity = np.array([list(map(float, line.strip().split())) for line in f])
#
# # 检查两个矩阵的尺寸是否一致
# assert semantic_similarity.shape == gaussian_similarity.shape, "两个矩阵的尺寸不一致！"
#
# # 融合矩阵
# fusion_matrix = np.where(gaussian_similarity > semantic_similarity, gaussian_similarity, semantic_similarity)
#
# # 保存融合矩阵
# np.savetxt('D:\Mamba\data\dis_fused_matrix.txt', fusion_matrix, fmt='%.6f')
#
# print("融合矩阵已保存到 D:\Mamba\data\dis_fused_matrix.txt")

# # 读取语义相似度矩阵
# with open('D:\Mamba\data\mfs.txt', 'r') as f:
#     semantic_similarity = np.array([list(map(float, line.strip().split())) for line in f])
#
# # 读取高斯核相似度矩阵
# with open('D:\Mamba\data\mg.txt', 'r') as f:
#     gaussian_similarity = np.array([list(map(float, line.strip().split())) for line in f])
#
# # 检查两个矩阵的尺寸是否一致
# assert semantic_similarity.shape == gaussian_similarity.shape, "两个矩阵的尺寸不一致！"
#
# # 融合矩阵
# fusion_matrix = np.where(gaussian_similarity > semantic_similarity, gaussian_similarity, semantic_similarity)
#
# # 保存融合矩阵
# np.savetxt('D:\Mamba\data\meta_fused_matrix.txt', fusion_matrix, fmt='%.6f')
#
# print("融合矩阵已保存到 D:\Mamba\data\meta_fused_matrix.txt")
#代谢物 1435 疾病242 药物3027 基因1877

# meta_matrix = np.loadtxt('D:\Mamba\data\meta_fused_matrix.txt')
# dis_matrix = np.loadtxt('D:\Mamba\data\dis_fused_matrix.txt')
# A_matrix = np.loadtxt('D:\Mamba\data/adjacency_matrix.txt')
# meta_gene_matrix = np.loadtxt('D:\Mamba\data\gene_meta_adjacency_matrix.txt')
# dis_drug_matrix = np.loadtxt('D:\Mamba\data\disease_drug_adjacency_matrix.txt')

meta_matrix = np.loadtxt('/root/autodl-tmp/data/meta_fused_matrix.txt')
dis_matrix = np.loadtxt('/root/autodl-tmp/data/dis_fused_matrix.txt')
A_matrix = np.loadtxt('/root/autodl-tmp/data/adjacency_matrix.txt')
meta_gene_matrix = np.loadtxt('/root/autodl-tmp/data/gene_meta_adjacency_matrix.txt')
dis_drug_matrix = np.loadtxt('/root/autodl-tmp/data/disease_drug_adjacency_matrix.txt')


# print(A_matrix.shape)
# print(meta_gene_matrix.shape)
# print(dis_drug_matrix.shape)


import networkx as nx
import random



G = nx.Graph()

# 添加代谢物和疾病之间的边
metabolite_count = A_matrix.shape[0]
disease_count = A_matrix.shape[1]


# 添加代谢物节点
for i in range(metabolite_count):
    G.add_node(f'metabolite_{i}', type='metabolite')

# 添加疾病节点
for i in range(disease_count):
    G.add_node(f'disease_{i}', type='disease')



for i in range(metabolite_count):
    for j in range(disease_count):
        if A_matrix[i, j] > 0:
            G.add_edge(f'metabolite_{i}', f'disease_{j}', weight=A_matrix[i, j])

# 添加代谢物和基因之间的边
gene_count = meta_gene_matrix.shape[1]
# 添加基因节点

for j in range(gene_count):
    G.add_node(f'gene_{j}', type='gene')

for i in range(metabolite_count):
    for j in range(gene_count):
        if meta_gene_matrix[i, j] > 0:
            G.add_edge(f'metabolite_{i}', f'gene_{j}', weight=meta_gene_matrix[i, j])

# 添加疾病和药物之间的边
drug_count = dis_drug_matrix.shape[1]
# 添加药物节点
for j in range(drug_count):
    G.add_node(f'drug_{j}', type='drug')

for i in range(disease_count):
    for j in range(drug_count):
        if dis_drug_matrix[i, j] > 0:
            G.add_edge(f'disease_{i}', f'drug_{j}', weight=dis_drug_matrix[i, j])

# 添加代谢物相似性边
for i in range(metabolite_count):
    for j in range(i + 1, metabolite_count):
        if meta_matrix[i, j] > 0:
            G.add_edge(f'metabolite_{i}', f'metabolite_{j}', weight=meta_matrix[i, j])

# 添加疾病相似性边
for i in range(disease_count):
    for j in range(i + 1, disease_count):
        if dis_matrix[i, j] > 0:
            G.add_edge(f'disease_{i}', f'disease_{j}', weight=dis_matrix[i, j])



# 设置节点类型
node_types = {}
for i in range(metabolite_count):
    node_types[f'metabolite_{i}'] = 'metabolite'
for i in range(disease_count):
    node_types[f'disease_{i}'] = 'disease'
for i in range(gene_count):
    node_types[f'gene_{i}'] = 'gene'
for i in range(drug_count):
    node_types[f'drug_{i}'] = 'drug'

nx.set_node_attributes(G, node_types, 'type')
# for node in G.nodes():
#     print(f"Node {node} is connected to:")
#     for neighbor, data in G[node].items():
#         print(f"Node {neighbor} with weight: {data['weight']}")

# 定义元路径，例如 Metabolite-Gene-Metabolite 或 Disease-Metabolite-Disease
meta_paths = [
    ['disease', 'metabolite'],
    ['drug','disease','metabolite'],
    ['gene','metabolite','disease'],
    ['disease','disease'],
    ['metabolite','metabolite']
    # ['disease', 'metabolite']
]

# 初始化并训练Metapath2Vec



import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from collections import defaultdict

def nx_to_torch_geometric(G):
    node_types = nx.get_node_attributes(G, 'type')
    node_type_dict = {'metabolite': 0, 'disease': 1, 'gene': 2, 'drug': 3}

    edge_index = []
    edge_attr = []
    for edge in G.edges(data=True):
        edge_index.append([edge[0], edge[1]])
        edge_attr.append([edge[2]['weight']])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    num_nodes = len(G.nodes)
    x = torch.zeros((num_nodes, len(node_type_dict)))
    for node, type_id in node_type_dict.items():
        for n in G.nodes:
            if node_types[n] == node:
                x[n] = torch.tensor([1 if i == type_id else 0 for i in range(len(node_type_dict))])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

data = nx_to_torch_geometric(G)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=128)

def train(model, data, epochs=50):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.x)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

train(model, data)

model.eval()
with torch.no_grad():
    node_embeddings = model(data.x, data.edge_index).numpy()

def save_embeddings(embeddings, file_path):
    with open(file_path, 'w') as f:
        for node_id, emb in enumerate(embeddings):
            emb_str = ' '.join(map(str, emb))
            f.write(f'{node_id} {emb_str}\n')

save_embeddings(node_embeddings, '/root/autodl-tmp/data/embeddings1.txt')

for i in range(5):
    print(f'Node: {i}, Embedding: {node_embeddings[i][:5]}...')



