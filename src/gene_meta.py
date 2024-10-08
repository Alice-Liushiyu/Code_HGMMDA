# 读取所有代谢物编号
with open('D:\Project\Mamba\data\metabolites.txt', 'r') as file:
    metabolites = [line.strip() for line in file]

# 读取所有基因编号
with open('D:\Project\Mamba\data\gene.txt', 'r') as file:
    genes = [line.strip() for line in file]

# 创建邻接矩阵并初始化为0
adj_matrix = [[0] * len(genes) for _ in range(len(metabolites))]

# 从第三个txt文件中读取代谢物和基因的关联关系，并更新邻接矩阵
with open('D:\Project\Mamba\data\hmdb_gene.txt', 'r') as file:
    for line in file:
        print(line)
        metabolite, gene = line.strip().split('\t')
        if metabolite in metabolites:
            metabolite_index = metabolites.index(metabolite)
            gene_index = genes.index(gene)
            adj_matrix[metabolite_index][gene_index] = 1

# 将邻接矩阵写入txt文件
with open('D:\Project\Mamba\data/gene_meta_adjacency_matrix.txt', 'w') as file:
    for row in adj_matrix:
        file.write('\t'.join(map(str, row)) + '\n')
