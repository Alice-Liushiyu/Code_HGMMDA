import numpy as np
import pandas as pd

# 从txt文件中加载代谢物的邻接矩阵
adjacency_matrix = np.loadtxt('D:\Project\Mamba\data/adjacency_matrix.txt', dtype=float)

print(type(adjacency_matrix))

# adjacency_matrix1 = [[adjacency_matrix[j][i] for j in range(len(adjacency_matrix))] for i in range(len(adjacency_matrix[0]))]
# adjacency_matrix1=np.array(adjacency_matrix1)
adjacency_matrix = adjacency_matrix.T
#计算高斯核相似性
def gaussian_kernel_similarity(x, y, sigma):
    return np.exp(-(np.linalg.norm(x - y) ** 2) * sigma)

# 计算相似性矩阵
num_disease = adjacency_matrix.shape[0]
print(num_disease)
similarity_matrix = np.zeros((num_disease, num_disease))

#
sum1 = 0
for k in range(num_disease):
   sum1 =  np.linalg.norm(adjacency_matrix[k])**2+sum1

sigma = 1/(sum1/(num_disease))
print(sigma)

for i in range(num_disease):
    for j in range(i,(num_disease)):
        similarity = gaussian_kernel_similarity(adjacency_matrix[i], adjacency_matrix[j], sigma)
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

# 将相似性矩阵保存到txt文件
np.savetxt('D:\Project\Mamba\data\dg.txt', similarity_matrix, fmt='%.6f', delimiter='\t')

print(np.exp(0))