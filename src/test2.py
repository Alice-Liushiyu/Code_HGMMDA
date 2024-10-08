
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
import numpy as np
import torch
import random
# import tensorflow as tf
#
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)
print('begin')

def read_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            label = parts[0]
            embedding = torch.tensor(list(map(float, parts[1:])), dtype=torch.float32)
            embeddings[label] = embedding
    return embeddings

def filter_embeddings(embeddings, categories):
    filtered_embeddings = {label: emb for label, emb in embeddings.items()
                           if any(label.startswith(cat) for cat in categories)}
    return filtered_embeddings

# 读取嵌入
# embeddings = read_embeddings("/root/autodl-tmp/data/fused_embeddings.txt")
# embeddings = read_embeddings("D:\Project\Mamba\experiment_data/fused_embeddings.txt")
# embeddings = read_embeddings("/root/autodl-tmp/data/embeddings.txt")
embeddings = read_embeddings("/root/autodl-tmp/data/fused_embeddings.txt")

# embeddings = read_embeddings("/root/autodl-tmp/experiment_data/no_mamba_fused_embeddings.txt")
# embeddings = read_embeddings("/root/autodl-tmp/experiment_data/no_outterupdate_fused_embeddings.txt")
# 过滤类别为 metabolite 和 disease 的嵌入
categories = ["metabolite", "disease"]
filtered_embeddings = filter_embeddings(embeddings, categories)

# print("过滤后的嵌入:")
# for label, embedding in filtered_embeddings.items():
#     print(f"Label: {label}, Embedding: {embedding}")
def compute_association_embeddings2(metabolite_embeddings, disease_embeddings):
    association_embeddings = {}
    for metab_label, metab_emb in metabolite_embeddings.items():
        for disease_label, disease_emb in disease_embeddings.items():
            association_label = f"{metab_label}_{disease_label}"
            # 假设 metab_emb 和 disease_emb 都是torch.Tensor，并且它们至少有一个维度是相同的
            # 这里我们简单地沿着最后一个维度（通常是特征维度）拼接它们
            # 如果你的张量是多维的，并且你想要沿着不同的维度拼接，可以调整dim参数
            association_embedding = torch.cat((metab_emb, disease_emb), dim=-1)
            association_embeddings[association_label] = association_embedding
    return association_embeddings

def compute_association_embeddings(metabolite_embeddings, disease_embeddings, weight1=0.7, weight2=0.3):
    association_embeddings = {}
    for metab_label, metab_emb in metabolite_embeddings.items():
        for disease_label, disease_emb in disease_embeddings.items():
            association_label = f"{metab_label}_{disease_label}"
            association_embedding = weight1 * metab_emb + weight2 * disease_emb
            association_embeddings[association_label] = association_embedding
    return association_embeddings

# 将嵌入按类别分开
metabolite_embeddings = {label: emb for label, emb in filtered_embeddings.items() if label.startswith("metabolite")}
disease_embeddings = {label: emb for label, emb in filtered_embeddings.items() if label.startswith("disease")}

# 计算代谢物和疾病的关联嵌入
association_embeddings = compute_association_embeddings2(metabolite_embeddings, disease_embeddings)
# print(association_embeddings)
# print(type(association_embeddings))
# print("代谢物和疾病的关联嵌入:")
# # for label, embedding in association_embeddings.items():
# #     print(f"Label: {label}, Embedding: {embedding}")
# print(len(association_embeddings))
# print(association_embeddings['metabolite_1_disease_1'].shape)


import numpy as np
# 读取标签数据
A_matrix = np.loadtxt('/root/autodl-tmp/data/adjacency_matrix.txt')
# A_matrix = np.loadtxt('D:\Project\Mamba\data/adjacency_matrix.txt')
# print(A_matrix.shape)
A_matrix = A_matrix.reshape(-1)
# print(A_matrix.shape)
# embedding_array = np.array(association_embeddings.values())

# embedding_array = np.array(list(association_embeddings.values()))

# 假设 association_embeddings 是一个字典
# association_embeddings = {'key1': value1, 'key2': value2, ...}

# 使用 keys() 方法获取字典的所有键
keys = list(association_embeddings.keys())
print(len(keys))

# 获取第一个键值对
first_key = keys[0]
first_value = association_embeddings[first_key]


array_size = [347270, 512]
embedding_array = np.zeros(array_size)
flag = 0
for i in keys:
    value = association_embeddings[i]
    arr = value.tolist()
    embedding_array[flag] = arr
    flag = flag+1

# 现在 first_key 和 first_value 分别包含字典的第一个键和值

# print(embedding_array.shape)


# 定义过采样器并设置所需的正负样本比例
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from collections import Counter


# 定义欠采样器并设置所需的正负样本比例
desired_ratio = 1  # 例如，设置正样本与负样本的比例为0.5
undersampler = RandomUnderSampler(sampling_strategy=desired_ratio,random_state=42)

# 进行欠采样
X_resampled, y_resampled = undersampler.fit_resample(embedding_array, A_matrix)
# print(X_resampled)
# print(y_resampled)
# # 输出原始数据集和欠采样后数据集的大小
# print(f'原始数据集大小: {embedding_array.shape[0]}')
# print(f'欠采样后数据集大小: {X_resampled.shape[0]}')
#
# # 统计并输出每个类别的样本数量
# counter = Counter(y_resampled)
# print(f'欠采样后每个类别的样本数量: {counter}')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, recall_score, f1_score, matthews_corrcoef, \
    precision_score, precision_recall_curve, average_precision_score

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 定义模型
class ConvNet(nn.Module):
    def __init__(self, input_shape):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Move model to device to ensure weights are on the same device as input
        self.to(device)

        # Now get the output size after convolution and pooling
        self.fc1 = nn.Linear(self._get_conv_output(input_shape), 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def _get_conv_output(self, shape):
        # 初始化的传递一个输入，通过卷积和池化层，以获取卷积后的数据尺寸
        x = torch.rand(1, 1, shape).to(device)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# 训练和评估模型
def train_and_evaluate(X, y, model, criterion, optimizer, train_loader, val_loader, epochs=50):
    # 将模型移动到 GPU 上
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            # 将数据移动到 GPU 上
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Print the loss for this epoch
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

    # 在验证集上评估
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            # 将数据移动到 GPU 上
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_pred_list.append(outputs.cpu().numpy())
    return np.concatenate(y_pred_list)

# 数据加载和训练
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
aucs, accuracies, f1s, precisions, recalls, mccs, auprs = [], [], [], [], [], [], []

# 假设 X 和 y 已经是 numpy 数组
X = X_resampled
y = y_resampled

for train_index, test_index in kfold.split(X, y):
    set_seed(42)
    # 数据划分
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 数据转化为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train.reshape(-1, 1, X_train.shape[1]), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.reshape(-1, 1, X_test.shape[1]), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = ConvNet(input_shape=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和预测
    y_pred = train_and_evaluate(X_train, y_train, model, criterion, optimizer, train_loader, val_loader, epochs=50)

    # 计算 AUC
    auc = roc_auc_score(y_test, y_pred)
    aucs.append(auc)

    # 计算不同阈值下的 TPR 和 FPR
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    J = tpr - fpr
    best_threshold = thresholds[np.argmax(J)]

    # 使用最佳阈值计算准确率
    y_pred_binary = (y_pred >= best_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    mcc = matthews_corrcoef(y_test, y_pred_binary)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    mccs.append(mcc)
    accuracies.append(accuracy)

    # 计算 AUPR
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred)
    auprs.append(aupr)
    print(f'Fold {len(aucs)}, AUC: {auc}, AUPR: {aupr}, Accuracy: {accuracy}')

# 输出所有折的评估指标
print('AUCs:', aucs)
print(f'Mean AUC: {np.mean(aucs)}')
print('AUPRs:', auprs)
print(f'Mean AUPR: {np.mean(auprs)}')
print('Precisions:', precisions)
print(f'Mean Precision: {np.mean(precisions)}')
print('Recalls:', recalls)
print(f'Mean Recall: {np.mean(recalls)}')
print('F1 scores:', f1s)
print(f'Mean F1: {np.mean(f1s)}')
print('MCCs:', mccs)
print(f'Mean MCC: {np.mean(mccs)}')
print('Accuracies:', accuracies)
print(f'Mean Accuracy: {np.mean(accuracies)}')




