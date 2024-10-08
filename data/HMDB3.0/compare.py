# 读取两个文件的内容
with open('D:\Project\Mamba\data\diseases.txt', 'r') as f1:
    data1 = f1.read().splitlines()

with open('D:\Project\Mamba\data\HMDB3.0\sp_disease-id.txt', 'r') as f2:
    data2 = f2.read().splitlines()

# 找出第一个文件中多出的编号
extra_indices_and_ids = [(index, id) for index, id in enumerate(data1) if id not in data2]

# 输出结果
for index, id in extra_indices_and_ids:
    print(f"Index: {index}, ID: {id}")
