#
# import csv
#
# # 读取第一个csv文件，获取药物编号和疾病编号
# drug_disease_info = []
# with open('D:\Project\Mamba\data\dis_drug.csv', 'r', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)  # 跳过表头
#     for row in reader:
#         if len(row) >= 4:  # 确保行中有足够的字段
#             drug_id, disease_name, disease_id, pubmed_ids = row[:4]  # 仅获取前四个字段
#             drug_disease_info.append((disease_id, drug_id))
#
# # 读取第二个txt文件，获取疾病编号
# disease_ids = []
# with open('D:\Project\Mamba\data\mesh.txt', 'r') as txtfile:
#     for line in txtfile:
#         disease_id = line.strip().split()[1]  # 获取第二列的疾病编号
#         disease_ids.append(disease_id)
#
# print(disease_ids)
#
# # 匹配疾病编号，获取匹配到的疾病与药物的对应关系
# matched_info = []
# for disease_id in disease_ids:
#     for d_id, dr_id in drug_disease_info:
#         print(d_id)
#         print(disease_id)
#         if d_id == disease_id:
#             matched_info.append((d_id, dr_id))
#
# # 将匹配到的疾病与药物的对应关系写入新的txt文件
# with open('D:\Project\Mamba\data\matched_drug_disease.txt', 'w') as outfile:
#     for disease_id, drug_id in matched_info:
#         outfile.write(f"{disease_id}\t{drug_id}\n")
#
# #读取第一个txt文件，获取疾病编号列表
# disease_mapping = {}
# with open('D:\Project\Mamba\data\mesh.txt', 'r') as file1:
#     for line in file1:
#         doid, new_doid = line.strip().split()
#         disease_mapping[new_doid] = doid
#
# print(disease_mapping)
#
# # 处理第二个txt文件，替换疾病编号并写入新文件
# with open('D:\Project\Mamba\data\matched_drug_disease_output.txt', 'r') as file2, open('D:\Project\Mamba\data\output.txt', 'w') as output_file:
#     for line in file2:
#         print(line)
#         old_doid, value = line.strip().split('\t', 1)  # 使用 '\t' 进行分割，仅分割一次
#         print(old_doid)
#         new_doid = disease_mapping.get(old_doid, old_doid)  # 获取对应的新疾病编号，如果不存在，则使用原始编号
#         output_file.write(f"{new_doid}\t{value}\n")



metabolites = []
# 读取所有代谢物编号
with open('D:\Project\Mamba\data\HMDB3.0\mesh_disease.txt', 'r') as file:
    for line in file:
        metabolites_id = line.strip().split()[0]  # 获取第二列的疾病编号
        metabolites.append(metabolites_id)

# 读取所有基因编号
with open('D:\Project\Mamba\data\drug_ids.txt', 'r') as file:
    genes = [line.strip() for line in file]

# 创建邻接矩阵并初始化为0
adj_matrix = [[0] * len(genes) for _ in range(len(metabolites))]

# 从第三个txt文件中读取代谢物和基因的关联关系，并更新邻接矩阵
with open('D:\Project\Mamba\data\output.txt', 'r') as file:
    for line in file:
        print(line)
        metabolite, gene = line.strip().split('\t')
        if metabolite in metabolites:
            metabolite_index = metabolites.index(metabolite)
            gene_index = genes.index(gene)
            adj_matrix[metabolite_index][gene_index] = 1

# 将邻接矩阵写入txt文件
with open('D:\Project\Mamba\data\HMDB3.0/disease_drug_adjacency_matrix.txt', 'w') as file:
    for row in adj_matrix:
        file.write(' '.join(map(str, row)) + '\n')

