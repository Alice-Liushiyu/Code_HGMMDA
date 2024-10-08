
import numpy as np
import networkx as nx
import random
from collections import defaultdict

import torch
from mamba_ssm import Mamba
# from src.mamba import ModelArgs, Mamba
# from src import mamba
# 设置全局随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# meta_matrix = np.loadtxt('D:\Mamba\data\meta_fused_matrix.txt')
# dis_matrix = np.loadtxt('D:\Mamba\data\dis_fused_matrix.txt')
# A_matrix = np.loadtxt('D:\Mamba\data/adjacency_matrix.txt')
# meta_gene_matrix = np.loadtxt('D:\Mamba\data\gene_meta_adjacency_matrix.txt')
# dis_drug_matrix = np.loadtxt('D:\Mamba\data\disease_drug_adjacency_matrix.txt')

# meta_matrix = np.loadtxt('/root/autodl-tmp/data/meta_fused_matrix.txt')
# dis_matrix = np.loadtxt('/root/autodl-tmp/data/dis_fused_matrix.txt')
# A_matrix = np.loadtxt('/root/autodl-tmp/data/adjacency_matrix.txt')
# meta_gene_matrix = np.loadtxt('/root/autodl-tmp/data/gene_meta_adjacency_matrix.txt')
# dis_drug_matrix = np.loadtxt('/root/autodl-tmp/data/disease_drug_adjacency_matrix.txt')

meta_matrix = np.loadtxt('/root/autodl-tmp/data/HMDB3.0/meta_fused_matrix.txt')
dis_matrix = np.loadtxt('/root/autodl-tmp/data/HMDB3.0/dis_fused_matrix.txt')
A_matrix = np.loadtxt('/root/autodl-tmp/data/HMDB3.0/dm.txt')
meta_gene_matrix = np.loadtxt('/root/autodl-tmp/data/HMDB3.0/gene_meta_adjacency_matrix.txt')
dis_drug_matrix = np.loadtxt('/root/autodl-tmp/data/HMDB3.0/disease_drug_adjacency_matrix.txt')

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

# 定义元路径，例如 Metabolite-Gene-Metabolite 或 Disease-Metabolite-Disease
# meta_paths = [
#     # ['disease', 'metabolite']
#     ['drug','disease','metabolite'],
#     ['gene','metabolite','disease'],
#     ['disease','disease'],
#     ['metabolite','metabolite']
#     # ['disease', 'metabolite']
# ]

import numpy as np

def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            node = values[0]
            vector = np.array([float(x) for x in values[1:]])
            embeddings[node] = vector
    return embeddings

# 使用示例
# file_path = '/root/autodl-tmp/data/embeddings.txt'
# file_path = '/root/autodl-tmp/data/embeddings.txt'
file_path = '/root/autodl-tmp/data/HMDB3.0/embeddings.txt'
embeddings = load_embeddings(file_path)




def load_sorted_nodes(file_path):
    node_groups = {
        'metabolite': {},
        'disease': {},
        'gene': {},
        'drug': {}
    }

    with open(file_path, 'r') as f:
        current_type = None
        for line in f:
            line = line.strip()
            if line.startswith("Node type:"):
                current_type = line.split(":")[1].strip()
            elif line:
                node, count = line.split()
                node_groups[current_type][node] = int(count)

    return node_groups

# 使用示例
# file_path = '/root/autodl-tmp/data/sorted_nodes.txt'
# file_path = '/root/autodl-tmp/data/sorted_nodes.txt'
file_path = '/root/autodl-tmp/data/HMDB3.0/sorted_nodes.txt'
node_groups = load_sorted_nodes(file_path)
print(type(node_groups['metabolite']))
# 初始化两个空列表来存储键和值

meta1 = []
meta1_embed = []
dis1 = []
dis1_embed = []
drug1 = []
drug1_embed = []
gene1 = []
gene1_embed = []
for key, value in node_groups['metabolite'].items():
    meta1.append(key)

for key, value in node_groups['disease'].items():
    dis1.append(key)

for key, value in node_groups['drug'].items():
    drug1.append(key)

for key, value in node_groups['gene'].items():
    gene1.append(key)

for i in meta1:
    meta1_embed.append(embeddings[i])

for i in dis1:
    dis1_embed.append(embeddings[i])

for i in drug1:
    drug1_embed.append(embeddings[i])

for i in gene1:
    gene1_embed.append(embeddings[i])

meta1_dict = {k: v for k, v in zip(meta1, meta1_embed)}
dis1_dict = {k: v for k, v in zip(dis1, dis1_embed)}
drug1_dict = {k: v for k, v in zip(drug1, drug1_embed)}
gene1_dict = {k: v for k, v in zip(gene1, gene1_embed)}

meta1_dicts = {label: torch.tensor(embedding, dtype=torch.float32) for label, embedding in meta1_dict.items()}
dis1_dicts = {label: torch.tensor(embedding, dtype=torch.float32) for label, embedding in dis1_dict.items()}
drug1_dicts = {label: torch.tensor(embedding, dtype=torch.float32) for label, embedding in drug1_dict.items()}
gene1_dicts = {label: torch.tensor(embedding, dtype=torch.float32) for label, embedding in gene1_dict.items()}

# 检查CUDA是否可用并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#metabolite
label_to_index1 = {label: i for i, label in enumerate(meta1_dicts.keys())}
index_to_label1 = {i: label for label, i in label_to_index1.items()}
# 将嵌入转换成张量 (假设所有嵌入的维度相同)
meta1_embeddings = torch.stack(list(meta1_dicts.values()))
print(meta1_embeddings.shape)
# 创建模型参数
args1 = Mamba(
    d_model=128,  # 嵌入维度
    d_state=16,
    expand=4,
    d_conv=4,
)
model1 = args1
model1 = model1.to(device)
input_ids1 = meta1_embeddings.unsqueeze(0)  # (1, num_labels)
print("mamba前")
print(input_ids1.shape)
# 假设 `input_ids1` 是你的输入张量
input_ids1 = input_ids1.to(device)
# 使用在正确设备上的输入调用模型
updated_embeddings1 = model1(input_ids1)  # (1, num_labels, d_model)
print("mamba后")
print(updated_embeddings1.shape)
# 去掉批次维度
updated_embeddings1 = updated_embeddings1.squeeze(0)
# 将更新后的嵌入保存到新的字典中
meta1_updated_label_embeddings = {index_to_label1[i]: updated_embeddings1[i] for i in range(len(updated_embeddings1))}

# 输出结果
# for label, embedding in meta1_updated_label_embeddings.items():
#     print(f"Label: {label}, Updated Embedding: {embedding}")

#disease
label_to_index2 = {label: i for i, label in enumerate(dis1_dicts.keys())}
index_to_label2 = {i: label for label, i in label_to_index2.items()}
# 将嵌入转换成张量 (假设所有嵌入的维度相同)
dis1_embeddings = torch.stack(list(dis1_dicts.values()))
# 创建模型参数
args2 = Mamba(
    d_model=128,  # 嵌入维度
    d_state=16,
    expand=4,
    d_conv=4,
)
# 初始化模型
model2 = args2
model2 = model2.to(device)
# 进行前向传播，得到更新后的嵌入
# 这里假设 input_ids 是标签的索引
input_ids2 = dis1_embeddings.unsqueeze(0)  # (1, num_labels)
input_ids2 = input_ids2.to(device)
updated_embeddings2 = model2(input_ids2)  # (1, num_labels, d_model)
# 去掉批次维度
updated_embeddings2 = updated_embeddings2.squeeze(0)
# 将更新后的嵌入保存到新的字典中
dis1_updated_label_embeddings = {index_to_label2[i]: updated_embeddings2[i] for i in range(len(updated_embeddings2))}
# 输出结果
# for label, embedding in dis1_updated_label_embeddings.items():
#     print(f"Label: {label}, Updated Embedding: {embedding}")

#drug
label_to_index3 = {label: i for i, label in enumerate(drug1_dicts.keys())}
index_to_label3 = {i: label for label, i in label_to_index3.items()}
# 将嵌入转换成张量 (假设所有嵌入的维度相同)
drug1_embeddings = torch.stack(list(drug1_dicts.values()))
# 创建模型参数
args3 = Mamba(
    d_model=128,  # 嵌入维度
    d_state=16,
    expand=4,
    d_conv=4,
)
# 初始化模型
model3 = args3
model3 = model3.to(device)
input_ids3 = drug1_embeddings.unsqueeze(0)  # (1, num_labels)
input_ids3 = input_ids3.to(device)
updated_embeddings3 = model3(input_ids3)  # (1, num_labels, d_model)
# 去掉批次维度
updated_embeddings3 = updated_embeddings3.squeeze(0)
# 将更新后的嵌入保存到新的字典中
drug1_updated_label_embeddings = {index_to_label3[i]: updated_embeddings3[i] for i in range(len(updated_embeddings3))}
# 输出结果
# for label, embedding in drug1_updated_label_embeddings.items():
#     print(f"Label: {label}, Updated Embedding: {embedding}")

#gene
label_to_index4 = {label: i for i, label in enumerate(gene1_dicts.keys())}
index_to_label4 = {i: label for label, i in label_to_index4.items()}
# 将嵌入转换成张量 (假设所有嵌入的维度相同)
gene1_embeddings = torch.stack(list(gene1_dicts.values()))
# 创建模型参数
args4 = Mamba(
    d_model=128,  # 嵌入维度
    d_state=16,
    expand=4,
    d_conv=4,
)
# 初始化模型
model4 = args4
model4 = model4.to(device)
# 进行前向传播，得到更新后的嵌入
# 这里假设 input_ids 是标签的索引
input_ids4 = gene1_embeddings.unsqueeze(0)  # (1, num_labels)
input_ids4 = input_ids4.to(device)

updated_embeddings4 = model4(input_ids4)  # (1, num_labels, d_model)
# 去掉批次维度
updated_embeddings4 = updated_embeddings4.squeeze(0)
# 将更新后的嵌入保存到新的字典中
gene1_updated_label_embeddings = {index_to_label4[i]: updated_embeddings4[i] for i in range(len(updated_embeddings4))}
# 输出结果
# for label, embedding in gene1_updated_label_embeddings.items():
#     print(f"Label: {label}, Updated Embedding: {embedding}")

import re
# 自定义排序函数
def sort_labels(labels):
    def extract_number(label):
        match = re.search(r'_(\d+)$', label)
        return int(match.group(1)) if match else -1
    return sorted(labels, key=extract_number)

# 使用排序函数对字典的标签进行排序
sorted_labels1 = sort_labels(list(meta1_updated_label_embeddings.keys()))
sorted_labels2 = sort_labels(list(dis1_updated_label_embeddings.keys()))
sorted_labels3 = sort_labels(list(drug1_updated_label_embeddings.keys()))
sorted_labels4 = sort_labels(list(gene1_updated_label_embeddings.keys()))


# 创建一个新的字典，按排序后的标签顺序组织
sorted_meta1_dict = {label: meta1_updated_label_embeddings[label] for label in sorted_labels1}
sorted_dis1_dict = {label: dis1_updated_label_embeddings[label] for label in sorted_labels2}
sorted_drug1_dict = {label: drug1_updated_label_embeddings[label] for label in sorted_labels3}
sorted_gene1_dict = {label: gene1_updated_label_embeddings[label] for label in sorted_labels4}

# # 输出排序后的字典
# print("排序后的字典:")
# for label, embedding in sorted_meta1_dict.items():
#     print(f"Label: {label}, Embedding: {embedding}")

merged_dict = {}
for d in [sorted_meta1_dict,sorted_dis1_dict, sorted_drug1_dict, sorted_gene1_dict]:
    merged_dict.update(d)









# vocab_size = 6581
# n_layer = 2
# d_model = 40
# model_args = ModelArgs(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size)


class NodeEmbeddingSorter:
    def __init__(self, graph, embeddings):
        self.graph = graph
        self.embeddings = embeddings

    def get_node_degrees(self):
        # 计算每个节点的度数
        node_degrees = {node: self.graph.degree(node) for node in self.graph.nodes()}
        return node_degrees

    def sort_nodes_by_degree(self, node_degrees):
        # 按照度数对节点进行排序
        sorted_nodes = sorted(node_degrees.items(), key=lambda item: item[1], reverse=True)
        return sorted_nodes

    def get_sorted_embeddings(self):
        node_degrees = self.get_node_degrees()
        sorted_nodes = self.sort_nodes_by_degree(node_degrees)

        # 创建一个新的字典来存储排序后的节点及其嵌入向量
        sorted_embeddings = {}
        for node, _ in sorted_nodes:
            if node in self.embeddings:
                sorted_embeddings[node] = self.embeddings[node]
            else:
                # 如果节点没有嵌入向量，则可以选择赋予一个默认值或跳过
                sorted_embeddings[node] = np.zeros(len(next(iter(self.embeddings.values()))))  # 这里假设嵌入维度一致

        return sorted_embeddings



# 实例化 NodeEmbeddingSorter
sorter = NodeEmbeddingSorter(G, embeddings)
sorted_embeddings = sorter.get_sorted_embeddings()
# print(sorted_embeddings)
sorted_embeddings = {label: torch.tensor(embedding, dtype=torch.float32) for label, embedding in sorted_embeddings.items()}



label_to_index = {label: i for i, label in enumerate(sorted_embeddings.keys())}
index_to_label = {i: label for label, i in label_to_index.items()}
# 将嵌入转换成张量 (假设所有嵌入的维度相同)
sorted_embeddings = torch.stack(list(sorted_embeddings.values()))

args = Mamba(
    d_model=128,  # 嵌入维度
    d_state=16,
    expand=4,
    d_conv=4,

)
# 初始化模型
model = args
model = model.to(device)
# 这里假设 input_ids 是标签的索引
input_ids = sorted_embeddings.unsqueeze(0)  # (1, num_labels)
input_ids = input_ids.to(device)
updated_embeddings = model(input_ids)  # (1, num_labels, d_model)
# 去掉批次维度
updated_embeddings = updated_embeddings.squeeze(0)
# 将更新后的嵌入保存到新的字典中
updated_label_embeddings = {index_to_label[i]: updated_embeddings[i] for i in range(len(updated_embeddings))}
# 输出结果
# for label, embedding in updated_label_embeddings.items():
#     print(f"Label: {label}, Updated Embedding: {embedding}")

# 自定义排序函数
def extract_number(label):
    match = re.search(r'_(\d+)$', label)
    return int(match.group(1)) if match else -1

# 按类型分组
grouped_dict = {
    "metabolite": {},
    "disease": {},
    "drug": {},
    "gene": {}
}

for key, value in updated_label_embeddings.items():
    if key.startswith("metabolite"):
        grouped_dict["metabolite"][key] = value
    elif key.startswith("disease"):
        grouped_dict["disease"][key] = value
    elif key.startswith("drug"):
        grouped_dict["drug"][key] = value
    elif key.startswith("gene"):
        grouped_dict["gene"][key] = value

# 对每个组内的键进行排序
sorted_grouped_dict = {category: dict(sorted(group.items(), key=lambda x: extract_number(x[0])))
                       for category, group in grouped_dict.items()}

# 合并所有组，按顺序存放
sorted_data_dict = {k: v for group in sorted_grouped_dict.values() for k, v in group.items()}

# 输出排序后的字典
# print("排序后的字典:")
# for label, embedding in sorted_data_dict.items():
#     print(f"Label: {label}, Embedding: {embedding}")

# # 打印结果
# for node, emb in sorted_embeddings.items():
#     print(f'Node: {node}, Embedding: {emb}')
# 拼接
fused_embedding_dict = {key: torch.cat((merged_dict[key], sorted_data_dict[key]), dim=0)
                        for key in merged_dict.keys()}

# 输出融合后的嵌入
# print("融合后的嵌入:")
# for label, embedding in fused_embedding_dict.items():
#     print(f"Label: {label}, Embedding: {embedding}")
# 加权平均
print(merged_dict['metabolite_0'].shape)
print(sorted_data_dict['metabolite_0'].shape)
print(len(meta1_dict))
print()

#
# 存储融合后的嵌入到txt文件中
with open("/root/autodl-tmp/data/HMDB3.0//fused_embeddings.txt", "w") as file:
    for label, embedding in fused_embedding_dict.items():
        embedding_str = " ".join(map(str, embedding.tolist()))
        file.write(f"{label} {embedding_str}\n")

