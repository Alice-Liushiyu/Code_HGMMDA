import numpy as np
#
# 读取邻接矩阵
def read_adjacency_matrix(file_path):
    adjacency_matrix = np.loadtxt(file_path)
    return adjacency_matrix

# 计算 Jaccard 相似度
def calculate_jaccard_similarity(adjacency_matrix):
    # 计算每对代谢物的 Jaccard 相似度
    num_metabolites = adjacency_matrix.shape[0]
    similarity_matrix = np.zeros((num_metabolites, num_metabolites))
    for i in range(num_metabolites):
        for j in range(num_metabolites):
            intersection = np.logical_and(adjacency_matrix[i], adjacency_matrix[j]).sum()
            union = np.logical_or(adjacency_matrix[i], adjacency_matrix[j]).sum()
            similarity_matrix[i][j] = intersection / union if union != 0 else 0
    return similarity_matrix

# 将相似性矩阵写入文本文件
def write_similarity_matrix(similarity_matrix, file_path):
    np.savetxt(file_path, similarity_matrix)
#
# 示例用法
adjacency_matrix = read_adjacency_matrix('D:\Project\Mamba\data\disease_drug_adjacency_matrix.txt')
similarity_matrix = calculate_jaccard_similarity(adjacency_matrix)
write_similarity_matrix(similarity_matrix, 'D:\Project\Mamba\data\ddj.txt')
# def read_adjacency_matrix(file_path):
#     adjacency_matrix = np.loadtxt(file_path)
#     return adjacency_matrix
# adjacency_matrix = read_adjacency_matrix('D:\Project\Mamba\data\mgj.txt')
# print(adjacency_matrix.shape)