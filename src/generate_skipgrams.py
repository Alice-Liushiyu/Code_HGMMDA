import pickle
import numpy as np
import networkx as nx
from keras_preprocessing.sequence import skipgrams

from src import metapath2vec
from src import multimetapath2vec
from src import node2vec
from src.data_encoding import to_integers


meta_matrix = np.loadtxt('D:\Mamba\data\meta_fused_matrix.txt')
dis_matrix = np.loadtxt('D:\Mamba\data\dis_fused_matrix.txt')
A_matrix = np.loadtxt('D:\Mamba\data/adjacency_matrix.txt')
meta_gene_matrix = np.loadtxt('D:\Mamba\data\gene_meta_adjacency_matrix.txt')
dis_drug_matrix = np.loadtxt('D:\Mamba\data\disease_drug_adjacency_matrix.txt')

# print(A_matrix.shape)
# print(meta_gene_matrix.shape)
# print(dis_drug_matrix.shape)




G = nx.Graph()

# 添加代谢物和疾病之间的边
metabolite_count = A_matrix.shape[0]
disease_count = A_matrix.shape[1]

for i in range(metabolite_count):
    for j in range(disease_count):
        if A_matrix[i, j] > 0:
            G.add_edge(f'metabolite_{i}', f'disease_{j}', weight=A_matrix[i, j])

# 添加代谢物和基因之间的边
gene_count = meta_gene_matrix.shape[1]

for i in range(metabolite_count):
    for j in range(gene_count):
        if meta_gene_matrix[i, j] > 0:
            G.add_edge(f'metabolite_{i}', f'gene_{j}', weight=meta_gene_matrix[i, j])

# 添加疾病和药物之间的边
drug_count = dis_drug_matrix.shape[1]

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

# 定义元路径，例如 Metabolite-Gene-Metabolite 或 Disease-Metabolite-Disease
meta_paths = [
    ["metabolite", "gene", "metabolite"],  # M-G-M
    ["disease", "metabolite", "disease"],  # D-M-D
    ["metabolite", "disease", "metabolite"],  # M-D-M
    ["disease", "drug", "disease"],  # D-D-D
    ["gene", "metabolite", "gene"],  # G-M-G
    ["drug", "disease", "drug"],  # Dr-D-Dr
]

def generate_skipgrams(algorithm):
    """
    Create dataset with skipgrams from random walking.

    :param algorithm: algorithm for random walking. Available values: 'node2vec', 'metapath2vec', 'multimetapath2vec'
    :param graph_filename: name of file containing graph
    """
    g = G

    num_walks = 10
    walk_length = 80


    walks = []
    if algorithm == "node2vec":
        ng = node2vec.Graph(g, is_directed=False, p=1., q=1.)
        ng.preprocess_transition_probs()
        walks = ng.simulate_walks(num_walks, walk_length)
    elif algorithm == "metapath2vec":
        walks = metapath2vec.Graph(g).simulate_walks(num_walks, walk_length, metapath=meta_paths)
    elif algorithm == "multimetapath2vec":
        walks = multimetapath2vec.Graph(g).simulate_walks(num_walks, walk_length, metapaths=
                                                                            [["JCH", "O", "NO", "O", "JCH"],
                                                                             ["JCH", "O", "WO", "O", "JCH"]])
    print("Encoding to integers")
    walks_encoded = to_integers(g.nodes, walks)

    print("Generating skipgrams")

    all_couples = []
    all_labels = []
    for walk_encoded in walks_encoded:
        couples, labels = skipgrams(sequence=walk_encoded, vocabulary_size=len(g.nodes) + 1)
        all_couples += couples
        all_labels += labels

    print(len(all_couples))
    print(len(all_labels))




if __name__ == '__main__':
    generate_skipgrams(algorithm="metapath2vec")
