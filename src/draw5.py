import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
import numpy as np
#no_inner
group1_fpr = np.loadtxt("D:\Project\Mamba\experiment_data/no_inner_fpr.txt")
group1_tpr = np.loadtxt("D:\Project\Mamba\experiment_data/no_inner_tpr.txt")
group1_precision = np.loadtxt("D:\Project\Mamba\experiment_data/no_inner_precision.txt")
group1_recall = np.loadtxt("D:\Project\Mamba\experiment_data/no_inner_recall.txt")

HGMMDA_fpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_fpr.txt")
HGMMDA_tpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_tpr.txt")
HGMMDA_precision = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_precision.txt")
HGMMDA_recall = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_recall.txt")
#no_outter
group2_fpr = np.loadtxt("D:\Project\Mamba\experiment_data/no_outter_fpr.txt")
group2_tpr = np.loadtxt("D:\Project\Mamba\experiment_data/no_outter_tpr.txt")
group2_precision = np.loadtxt("D:\Project\Mamba\experiment_data/no_outter_precision.txt")
group2_recall = np.loadtxt("D:\Project\Mamba\experiment_data/no_outter_recall.txt")
#no_sort
group3_fpr = np.loadtxt("D:\Project\Mamba\experiment_data/no_sort_fpr.txt")
group3_tpr = np.loadtxt("D:\Project\Mamba\experiment_data/no_sort_tpr.txt")
group3_precision = np.loadtxt("D:\Project\Mamba\experiment_data/no_sort_precision.txt")
group3_recall = np.loadtxt("D:\Project\Mamba\experiment_data/no_sort_recall.txt")


# fpr1=[]
# for i in flod2_fpr:
#     if i


# plt.plot(group1_fpr, group1_tpr, label=f'Group1 AUC = 0.9727', color='b',linewidth=1)  # 绘制ROC曲线，标注AUC的值{auc_score:0.9642}
# plt.plot(group2_fpr, group2_tpr, label=f'Group2 AUC = 0.9726',color='y',linewidth=1)
# plt.plot(group3_fpr, group3_tpr, label=f'Group3 AUC = 0.9724',color='g',linewidth=1)
# plt.plot(HGMMDA_fpr, HGMMDA_tpr, label=f'HGMMDA AUC = 0.9778',color='r',linewidth=1)
#
#
#
# # 随即分类器没有分类能力，其FPR=TPR。随机分类器的性能通常表示为ROC曲线上的对角线
# plt.plot([0, 1], [0, 1], linestyle='--', color='navy')  # 绘制随机分类器的ROC曲线
# plt.xlim(0, 0.40)
# plt.ylim(0.75, 1)
# plt.xlabel('False Positive Rate')  # x轴标签为FPR
# plt.ylabel('True Positive Rate')   # y轴标签为TPR
# plt.title('ROC Curve')             # 设置标题
# plt.legend()
# plt.savefig('D:\Project\Mamba\experiment_data/消融比较消融AUC.png')
# plt.show()


plt.plot(group1_recall, group1_precision, label=f'Group1 AUPR = 0.9680', color='b',linewidth=1)  # 绘制ROC曲线，标注AUC的值{auc_score:0.9642}
plt.plot(group2_recall, group2_precision, label=f'Group2 AUPR = 0.9672',color='y',linewidth=1)
plt.plot(group3_recall, group3_precision, label=f'Group3 AUPR = 0.9675',color='g',linewidth=1)
plt.plot(HGMMDA_recall, HGMMDA_precision, label=f'HGMMDA AUPR = 0.9730',color='r',linewidth=1)
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