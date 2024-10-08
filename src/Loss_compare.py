
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# # 禁用 XLA 编译（如果启用了）
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
# os.environ['PYTHONHASHSEED'] = str(42)
import numpy as np
import torch
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
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
# embeddings = read_embeddings("D:\Project\Mamba\data/fused_embeddings.txt")
embeddings = read_embeddings("D:\Project\Mamba\data/fused_embeddings.txt")

# embeddings = read_embeddings("/root/autodl-tmp/data/embeddings.txt")

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

def compute_association_embeddings(metabolite_embeddings, disease_embeddings, weight1=0.5, weight2=0.5):
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
# A_matrix = np.loadtxt('/root/autodl-tmp/data/adjacency_matrix.txt')

A_matrix = np.loadtxt('D:\Project\Mamba\data/adjacency_matrix.txt')

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
print(X_resampled)
# # 输出原始数据集和欠采样后数据集的大小
# print(f'原始数据集大小: {embedding_array.shape[0]}')
# print(f'欠采样后数据集大小: {X_resampled.shape[0]}')
#
# # 统计并输出每个类别的样本数量
# counter = Counter(y_resampled)
# print(f'欠采样后每个类别的样本数量: {counter}')


import torch
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, recall_score, f1_score, matthews_corrcoef, \
    precision_score, precision_recall_curve, average_precision_score

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def create_model(input_shape,lr):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold = KFold(n_splits=5, shuffle=True,random_state=42)
aucs = []
accuracies = []
f1s = []
precisions = []
recalls = []
mccs = []
auprs = []
X = X_resampled
y = y_resampled
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
epoch_losses = []
for lr in learning_rates:
    # 存储每个学习率下的损失值
    losses = []
    for train_index, test_index in kfold.split(X, y):
        fold_losses = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train)
        model = create_model((X_train.shape[1], 1),lr)  # 假设每个样本是8000x1的数组
        history = model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=200, batch_size=32)
        fold_losses.extend(history.history['loss'])
        # 将每折的损失值存储到总的损失值数组中
        epoch_losses.append(fold_losses)
        print(epoch_losses)
        # 计算每个 epoch 的平均损失值
    mean_epoch_losses = np.mean(epoch_losses, axis=0)
    print("每个 epoch 的平均损失值：", mean_epoch_losses)
    np.savetxt('D:\Project\Mamba\experiment_data\HGMMDA_' + str(lr) + '_loss.txt', mean_epoch_losses)
        # y_pred = model.predict(X_test.reshape(-1, X_test.shape[1], 1))
        # y_pred = y_pred.ravel()
        # auc = roc_auc_score(y_test, y_pred)
        # aucs.append(auc)
        # # Convert predictions to binary (0 or 1) using a threshold of 0.5
        # # 计算不同阈值下的 TPR 和 FPR
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        #
        # # Youden's J statistic 最大化
        # J = tpr - fpr
        # best_threshold = thresholds[np.argmax(J)]
        #
        # # 使用最佳阈值计算准确率
        # y_pred_binary = (y_pred >= best_threshold).astype(int)
        # accuracy = accuracy_score(y_test, y_pred_binary)
        # print(f'Best Threshold: {best_threshold}, Accuracy: {accuracy}')
        # precision = precision_score(y_test, y_pred_binary)
        # recall = recall_score(y_test, y_pred_binary)
        # f1 = f1_score(y_test, y_pred_binary)
        # mcc = matthews_corrcoef(y_test, y_pred_binary)
        #
        # precisions.append(precision)
        # recalls.append(recall)
        # f1s.append(f1)
        # mccs.append(mcc)
        # # Calculate accuracy
        # accuracies.append(accuracy)
        # precisions.append(precision)
        # recalls.append(recall)
        # f1s.append(f1)
        # mccs.append(mcc)
        # accuracies.append(accuracy)
        #
        # # Calculate AUPR
        # precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
        # aupr = average_precision_score(y_test, y_pred)
        # auprs.append(aupr)
        #
        # print(f'Best Threshold: {best_threshold}, Accuracy: {accuracy}')
        # print(f'Fold {len(aucs)}, AUC: {auc}, AUPR: {aupr}, Accuracy: {accuracy}')
        #
    # # 输出所有折的评估指标
    # print('AUCs:', aucs)
    # print(f'Mean AUC: {np.mean(aucs)}')
    # print('AUPRs:', auprs)
    # print(f'Mean AUPR: {np.mean(auprs)}')
    # print('Precisions:', precisions)
    # print(f'Mean Precision: {np.mean(precisions)}')
    # print('Recalls:', recalls)
    # print(f'Mean Recall: {np.mean(recalls)}')
    # print('F1 scores:', f1s)
    # print(f'Mean F1: {np.mean(f1s)}')
    # print('MCCs:', mccs)
    # print(f'Mean MCC: {np.mean(mccs)}')
    # print('Accuracies:', accuracies)
    # print(f'Mean Accuracy: {np.mean(accuracies)}')

