import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
import numpy as np

flod2_fpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_2kflod_fpr.txt")
flod2_tpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_2kflod_tpr.txt")
flod2_precision = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_2kflod_precision.txt")
flod2_recall = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_2kflod_recall.txt")

flod5_fpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_fpr.txt")
flod5_tpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_tpr.txt")
flod5_precision = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_precision.txt")
flod5_recall = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_5kflod_recall.txt")

flod10_fpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_10kflod_fpr.txt")
flod10_tpr = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_10kflod_tpr.txt")
flod10_precision = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_10kflod_precision.txt")
flod10_recall = np.loadtxt("D:\Project\Mamba\experiment_data\HGMMDA_10kflod_recall.txt")


# fpr1=[]
# for i in flod2_fpr:
#     if i

plt.plot(flod2_fpr, flod2_tpr, label=f'2-Flod AUC = 0.9701', color='b',linewidth=1)  # 绘制ROC曲线，标注AUC的值{auc_score:0.9642}
plt.plot(flod5_fpr, flod5_tpr, label=f'5-Flod AUC = 0.9778',color='r',linewidth=1)
plt.plot(flod10_fpr, flod10_tpr, label=f'10-Flod AUC = 0.9773',color='g',linewidth=1)


# 随即分类器没有分类能力，其FPR=TPR。随机分类器的性能通常表示为ROC曲线上的对角线
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')  # 绘制随机分类器的ROC曲线
plt.xlim(0, 0.40)
plt.ylim(0.75, 1)
plt.xlabel('False Positive Rate')  # x轴标签为FPR
plt.ylabel('True Positive Rate')   # y轴标签为TPR
plt.title('ROC Curve')             # 设置标题
plt.legend()
plt.savefig('D:\Project\Mamba\experiment_data/模型局部比较AUC.png')
plt.show()


# plt.plot(flod2_recall, flod2_precision, label=f'2-Flod AUPR = 0.9637', color='b',linewidth=1)  # 绘制ROC曲线，标注AUC的值{auc_score:0.9642}
# plt.plot(flod5_recall, flod5_precision, label=f'5-Flod AUPR = 0.9730',color='r',linewidth=1)
# plt.plot(flod10_recall, flod10_precision, label=f'10-Flod AUPR = 0.9720',color='g',linewidth=1)
# plt.xlim(0.75, 1)
# plt.ylim(0.75, 1)
# # 随即分类器没有分类能力，其FPR=TPR。随机分类器的性能通常表示为ROC曲线上的对角线
# # plt.plot([0, 1], [0, 1], linestyle='--', color='navy')  # 绘制随机分类器的ROC曲线
# plt.xlabel('Recall')  # x轴标签为FPR
# plt.ylabel('Precision')   # y轴标签为TPR
# plt.title('Cross Validation PR Curve')             # 设置标题
# plt.legend(loc='lower left')
# # plt.show()
# plt.savefig('D:\Project\Mamba\experiment_data/模型局部比较AUPR.png')
# plt.show()
