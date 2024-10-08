import torch
import numpy as np
from gensim.models import Word2Vec
import numpy as np
import random
import networkx as nx
from collections import defaultdict
print('begin')
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)
# # 读取语义相似度矩阵
# with open('D:\Project\Mamba\data\HMDB3.0\DisSim.txt', 'r') as f:
#     semantic_similarity = np.array([list(map(float, line.strip().split())) for line in f])
#
# # 读取高斯核相似度矩阵
# with open('D:\Project\Mamba\data\HMDB3.0\dk.txt', 'r') as f:
#     gaussian_similarity = np.array([list(map(float, line.strip().split())) for line in f])
#
# # 检查两个矩阵的尺寸是否一致
# assert semantic_similarity.shape == gaussian_similarity.shape, "两个矩阵的尺寸不一致！"
#
# # 融合矩阵
# fusion_matrix = np.where(gaussian_similarity > semantic_similarity, gaussian_similarity, semantic_similarity)
#
# # 保存融合矩阵
# np.savetxt('D:\Project\Mamba\data\HMDB3.0\dis_fused_matrix.txt', fusion_matrix, fmt='%.6f')
#
# print("融合矩阵已保存到 D:\Project\Mamba\data\HMDB3.0\dis_fused_matrix.txt")
#
# # 读取语义相似度矩阵
# with open('D:\Project\Mamba\data\HMDB3.0\mfs.txt', 'r') as f:
#     semantic_similarity = np.array([list(map(float, line.strip().split())) for line in f])
#
# # 读取高斯核相似度矩阵
# with open('D:\Project\Mamba\data\HMDB3.0\mk.txt', 'r') as f:
#     gaussian_similarity = np.array([list(map(float, line.strip().split())) for line in f])
#
# # 检查两个矩阵的尺寸是否一致
# assert semantic_similarity.shape == gaussian_similarity.shape, "两个矩阵的尺寸不一致！"
#
# # 融合矩阵
# fusion_matrix = np.where(gaussian_similarity > semantic_similarity, gaussian_similarity, semantic_similarity)
#
# # 保存融合矩阵
# np.savetxt('D:\Project\Mamba\data\HMDB3.0\meta_fused_matrix.txt', fusion_matrix, fmt='%.6f')
#
# print("融合矩阵已保存到 D:\Project\Mamba\data\HMDB3.0\meta_fused_matrix.txt")
# 代谢物 1435 疾病242 药物3027 基因1877

meta_matrix = np.loadtxt('D:\Project\Mamba\data\HMDB3.0\meta_fused_matrix.txt')
dis_matrix = np.loadtxt('D:\Project\Mamba\data\HMDB3.0\dis_fused_matrix.txt')
A_matrix = np.loadtxt('D:\Project\Mamba\data\HMDB3.0/dm.txt')
meta_gene_matrix = np.loadtxt('D:\Project\Mamba\data\HMDB3.0\gene_meta_adjacency_matrix.txt')
dis_drug_matrix = np.loadtxt('D:\Project\Mamba\data\HMDB3.0\disease_drug_adjacency_matrix.txt')

# meta_matrix = np.loadtxt('/root/autodl-tmp/data/meta_fused_matrix.txt')
# dis_matrix = np.loadtxt('/root/autodl-tmp/data/dis_fused_matrix.txt')
# A_matrix = np.loadtxt('/root/autodl-tmp/data/adjacency_matrix.txt')
# meta_gene_matrix = np.loadtxt('/root/autodl-tmp/data/gene_meta_adjacency_matrix.txt')
# dis_drug_matrix = np.loadtxt('/root/autodl-tmp/data/disease_drug_adjacency_matrix.txt')


print(A_matrix.shape)
print(meta_gene_matrix.shape)
print(dis_drug_matrix.shape)
#(1435, 177)
#(1435, 1877)
#(177, 3027)

import networkx as nx
import random

class Metapath2Vec:
    def __init__(self, graph, meta_paths, walk_length, num_walks, embedding_dim, window_size, workers, epochs,seed=None):
        self.node_walk_counts = defaultdict(int)
        self.graph = graph
        self.meta_paths = meta_paths
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.workers = workers
        self.epochs = epochs
        self.embeddings = None
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    def random_walk(self, start_node, meta_path):
        walk = [start_node]
        current_node = start_node
        current_type_index = 1

        while len(walk) < self.walk_length:
            current_type = meta_path[current_type_index % len(meta_path)]
            neighbors = [node for node in self.graph.neighbors(current_node) if self.graph.nodes[node]['type'] == current_type]
            if len(neighbors) == 0:
                break
            # walk.append(current_node)
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
            current_type_index += 1
        return walk

    def generate_walks(self):
        walks = []
        for meta_path in self.meta_paths:
            for _ in range(self.num_walks):
                nodes = [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] == meta_path[0]]
                # print(nodes)
                random.shuffle(nodes)
                for node in nodes:
                    walk = self.random_walk(node, meta_path)
                    # print(len(walk))
                    if len(walk)>1:
                        for w in walk:
                            self.node_walk_counts[w] += 1
                    walks.append(walk)
                    # print(walks)
        return walks

    def train(self):
        walks = self.generate_walks()
        model = Word2Vec(sentences=walks, vector_size=self.embedding_dim, window=self.window_size, sg=1, workers=self.workers, epochs=self.epochs)
        self.embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        return self.embeddings
    def save_embeddings(self, file_path):
        with open(file_path, 'w') as f:
            all_nodes = sorted(self.graph.nodes(), key=lambda x: (x.split('_')[0], int(x.split('_')[1])))
            embedding_dim = len(next(iter(self.embeddings.values())))

            # Calculate average embedding for missing nodes
            avg_embedding = np.mean(list(self.embeddings.values()), axis=0)

            for node in all_nodes:
                if node in self.embeddings:
                    emb_str = ' '.join(map(str, self.embeddings[node]))
                else:
                    emb_str = ' '.join(map(str, avg_embedding))
                f.write(f'{node} {emb_str}\n')
    def save_sorted_nodes(self, file_path):
        node_groups = defaultdict(list)
        for node in self.graph.nodes():
            node_type = node.split('_')[0]
            node_groups[node_type].append(node)

        for node_type in node_groups.keys():
            node_groups[node_type] = sorted(node_groups[node_type], key=lambda x: (self.node_walk_counts.get(x, 0), int(x.split('_')[1])))

        all_nodes = sorted(self.graph.nodes(), key=lambda x: (x.split('_')[0], int(x.split('_')[1])))

        with open(file_path, 'w') as f:
            for node_type, nodes in node_groups.items():
                f.write(f"Node type: {node_type}\n")
                for node in nodes:
                    walk_count = self.node_walk_counts.get(node, 0)
                    f.write(f"{node} {walk_count}\n")
                # Adding nodes that are not in walks
                for node in all_nodes:
                    if node.split('_')[0] == node_type and node not in nodes:
                        f.write(f"{node} 0\n")
                f.write("\n")
    # def save_embeddings(self, file_path):
    #     with open(file_path, 'w') as f:
    #         for node in sorted(self.embeddings.keys(), key=lambda x: (x.split('_')[0], int(x.split('_')[1]))):
    #             emb_str = ' '.join(map(str, self.embeddings[node]))
    #             f.write(f'{node} {emb_str}\n')

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
    # ['disease', 'metabolite'],
    # ['drug','disease','metabolite'],
    # ['gene','metabolite','disease'],
    ['disease','disease'],
    ['metabolite','metabolite'],
    ['disease',"metabolite", "gene", "metabolite",'disease'],  # M-G-M
    ["disease", "metabolite", "disease"],  # D-M-D
    ["metabolite", "disease", "metabolite"],  # M-D-M
    ["metabolite", "disease", "drug", "disease", "metabolite"],  # D-D-D
    # ["gene", "metabolite", "gene"],  # G-M-G
    # ["drug", "disease", "drug"],  # Dr-D-Dr
]

# 初始化并训练Metapath2Vec
mp2v = Metapath2Vec(G, meta_paths, walk_length=8, num_walks=10, embedding_dim=128, window_size=5, workers=4, epochs=10,seed=42)
# mp2v = Metapath2Vec(G, meta_paths, walk_length=10, num_walks=10, embedding_dim=128, window_size=5, workers=4, epochs=20,seed=42)

# mp2v = Metapath2Vec(G, meta_paths, walk_length=20, num_walks=20, embedding_dim=128, window_size=5, workers=4, epochs=50,seed=42)
embeddings = mp2v.train()
# print(embeddings.items())

mp2v.save_embeddings('D:\Project\Mamba\data\HMDB3.0/embeddings.txt')
mp2v.save_sorted_nodes('D:\Project\Mamba\data\HMDB3.0/sorted_nodes.txt')
# 打印部分嵌入向量
for node, emb in list(embeddings.items())[:5]:  # 仅打印前5个节点
    print(f'Node: {node}, Embedding: {emb[:5]}...')  # 仅打印前5个维度




