#HMDB3.0 VS HMDB5.0
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
import numpy as np
#no_inner
HMDB3_fpr = np.loadtxt("D:\Project\Mamba\data\HMDB3.0\HMDB3_fpr.txt")
HMDB3_tpr = np.loadtxt("D:\Project\Mamba\data\HMDB3.0\HMDB3_tpr.txt")
HMDB3_precision = np.loadtxt("D:\Project\Mamba\data\HMDB3.0\HMDB3_precision.txt")
HMDB3_recall = np.loadtxt("D:\Project\Mamba\data\HMDB3.0\HMDB3_recall.txt")

HMDB5_fpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_fpr.txt")
HMDB5_tpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_tpr.txt")
HMDB5_precision = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_precision.txt")
HMDB5_recall = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_recall.txt")



# fpr1=[]
# for i in flod2_fpr:
#     if i

# plt.plot(HMDB3_fpr, HMDB3_tpr, label=f'HMDB3.0 AUC = 0.9761',color='g',linewidth=1)
# plt.plot(HMDB5_fpr, HMDB5_tpr, label=f'HMDB5.0 AUC = 0.9778',color='r',linewidth=1)
#
#
#
# # 随即分类器没有分类能力，其FPR=TPR。随机分类器的性能通常表示为ROC曲线上的对角线
# plt.plot([0, 1], [0, 1], linestyle='--', color='navy')  # 绘制随机分类器的ROC曲线
# # plt.xlim(0, 0.40)
# # plt.ylim(0.75, 1)
# plt.xlabel('False Positive Rate')  # x轴标签为FPR
# plt.ylabel('True Positive Rate')   # y轴标签为TPR
# plt.title('ROC Curve')             # 设置标题
# plt.legend()
# plt.savefig('D:\Project\Mamba\data\HMDB3.0\数据集对比AUC.png')
# plt.show()


plt.plot(HMDB3_recall, HMDB3_precision, label=f'Group1 AUPR = 0.9680', color='b',linewidth=1)  # 绘制ROC曲线，标注AUC的值{auc_score:0.9642}
plt.plot(HMDB3_recall, HMDB3_precision, label=f'Group2 AUPR = 0.9672',color='y',linewidth=1)

plt.xlim(0.75, 1)
plt.ylim(0.75, 1)
# 随即分类器没有分类能力，其FPR=TPR。随机分类器的性能通常表示为ROC曲线上的对角线
# plt.plot([0, 1], [0, 1], linestyle='--', color='navy')  # 绘制随机分类器的ROC曲线
plt.xlabel('Recall')  # x轴标签为FPR
plt.ylabel('Precision')   # y轴标签为TPR
plt.title('Cross Validation PR Curve')             # 设置标题
plt.legend(loc='lower left')

plt.savefig('D:\Project\Mamba\experiment_data/消融局部比较AUPR.png')
plt.show()