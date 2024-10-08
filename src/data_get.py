import numpy as np

# 读取关联数据文件，生成字典
data = {}
with open("D:\Project\Mamba\data/alldata.txt", "r") as f:
    for line in f:
        metabolite, disease, relation = line.strip().split()
        if metabolite not in data:
            data[metabolite] = {}
        data[metabolite][disease] = int(relation)

def read_order(file_path):
    with open(file_path, "r") as file:
        order = [line.strip() for line in file]
    return order

metabolites = read_order("D:\Project\Mamba\data/metabolites.txt")
diseases = read_order("D:\Project\Mamba\data/diseases.txt")




# 生成 0/1 矩阵
matrix = np.zeros((len(metabolites), len(diseases)), dtype=int)
for i, metabolite in enumerate(metabolites):
    for j, disease in enumerate(diseases):
        if metabolite in data and disease in data[metabolite]:
            matrix[i, j] = data[metabolite][disease]

# 存储到 txt 文件
with open("D:\Project\Mamba\data/adjacency_matrix.txt", "w") as f:
    # 写入列顺序
    # 写入矩阵数据
    for i, row in enumerate(matrix):
        f.write("\t".join(map(str, row)) + "\n")
