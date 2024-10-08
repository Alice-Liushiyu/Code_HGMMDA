
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from matplotlib import pyplot as plt

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# # 禁用 XLA 编译（如果启用了）

import numpy as np
import torch
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras import mixed_precision

# 设置全局浮点精度为 float32
mixed_precision.set_global_policy('float32')

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
embeddings = read_embeddings("D:\Project\Mamba\data\HMDB3.0/fused_embeddings.txt")

# embeddings = read_embeddings("D:\Project\Mamba\experiment_data/fused_embeddings.txt")

# embeddings = read_embeddings("/root/autodl-tmp/data/embeddings.txt")

# embeddings = read_embeddings("D:\Project\Mamba\experiment_data/no_mamba_fused_embeddings.txt")
# embeddings = read_embeddings("D:\Project\Mamba\experiment_data/no_innerupdate_fused_embeddings.txt")
# embeddings = read_embeddings("D:\Project\Mamba\experiment_data/no_outterupdate_fused_embeddings.txt")

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

# A_matrix = np.loadtxt('D:\Project\Mamba\data/adjacency_matrix.txt')
A_matrix = np.loadtxt('D:\Project\Mamba\data\HMDB3.0\dm.txt')
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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, recall_score, f1_score, matthews_corrcoef, \
    precision_score, precision_recall_curve, average_precision_score

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def create_model(input_shape):
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold = KFold(n_splits=5, shuffle=True,random_state=42)
aucs = []
accuracies = []
f1s = []
precisions = []
recalls = []
mccs = []
auprs = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)

precisions_curve = []
mean_recall = np.linspace(0, 1, 100)
X = X_resampled
y = y_resampled
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
for train_index, test_index in kfold.split(X, y):
    print(train_index)
    print(test_index)
all_preds = []
# 固定数据集分割和模型训练的随机性
for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
    # 打印索引以确保每次运行时相同
    print(f"Fold {fold}")
    print(train_index)
    print(test_index)

    # 使用固定种子确保数据顺序一致
    set_seed(seed)
    np.random.shuffle(train_index)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 固定随机性
    set_seed(seed)

    # 创建和编译模型
    model = create_model((X_train.shape[1], 1))

    # 保存初始权重
    initial_weights = model.get_weights()

    # 每次都加载相同的初始权重
    model.set_weights(initial_weights)

    # 训练模型
    model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=20, batch_size=32,
              validation_data=(X_test.reshape(-1, X_test.shape[1], 1), y_test))
    # model.save_weights(f'D:\Project\Mamba\exm_analysis\model_fold_{train_index[0]}.h5')  # 保存权重文件，使用唯一文件名
    # 在预测前加载保存的模型权重
    # model.load_weights(f'model_fold_{len(aucs)}.h5')
    y_pred = model.predict(X_test.reshape(-1, X_test.shape[1], 1),batch_size=1)
    print(y_pred)
    y_pred = y_pred.ravel()
    # preds = model.predict(embedding_array.reshape(-1, embedding_array.shape[1], 1))
    # print(preds)

    # 存储预测结果
    # all_preds.append(preds)
    auc = roc_auc_score(y_test, y_pred)
    aucs.append(auc)
    # Convert predictions to binary (0 or 1) using a threshold of 0.5
    # 计算不同阈值下的 TPR 和 FPR
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Youden's J statistic 最大化
    J = tpr - fpr
    best_threshold = thresholds[np.argmax(J)]

    # 使用最佳阈值计算准确率
    y_pred_binary = (y_pred >= best_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f'Best Threshold: {best_threshold}, Accuracy: {accuracy}')
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    mcc = matthews_corrcoef(y_test, y_pred_binary)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    mccs.append(mcc)
    # Calculate accuracy
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    mccs.append(mcc)
    accuracies.append(accuracy)

    # Calculate AUPR
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred)
    auprs.append(aupr)
    print(f'Fold {len(aucs)}, AUC: {auc}, AUPR: {aupr}, Accuracy: {accuracy}')
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred)
    auprs.append(aupr)
    precisions_curve.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
    precisions_curve[-1][0] = 1.0  # 将开始点设置为1


# average_preds = np.mean(all_preds, axis=0)
# average_preds=average_preds.reshape(1435,242)
# print(average_preds.shape)
# np.savetxt('D:\Project\Mamba\exm_analysis\predict2.txt', average_preds)

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
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
mean_aupr = np.mean(auprs)
mean_precision = np.mean(precisions_curve, axis=0)
# 绘制AUC曲线
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

np.savetxt('D:\Project\Mamba\data\HMDB3.0/HMDB3_fpr.txt', mean_fpr)
np.savetxt('D:\Project\Mamba\data\HMDB3.0/HMDB3_tpr.txt', mean_tpr)
#
#
# 绘制AUPR曲线
plt.subplot(1, 2, 2)
plt.plot(mean_recall, mean_precision, color='r', label=f'Mean PR (AUPR = {mean_aupr:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Mean PR Curve')
plt.legend(loc='lower left')
plt.grid(True)
np.savetxt('D:\Project\Mamba\data\HMDB3.0/HMDB3_precision.txt', mean_precision)
np.savetxt('D:\Project\Mamba\data\HMDB3.0/HMDB3_recall.txt', mean_recall)

plt.tight_layout()
plt.show()
