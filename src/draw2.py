import matplotlib.pyplot as plt
import seaborn as sns

# 数据
dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
mean_auc = [
    0.9702173527590853, 0.9758704869711515, 0.9772681050719608,
    0.9762793701650094, 0.9778453183088576, 0.9764577087499638,
    0.9758127386516051, 0.9741940009795365, 0.8769325595615198, 0.8744738610723892
]
mean_aupr = [
    0.9632579862685912, 0.9698389045171456, 0.9715487852319373,
    0.9710529304042101, 0.9730164112499228, 0.9712401887057502,
    0.9707189558570327, 0.9691243862740901, 0.8741389464820998, 0.87271043719992
]

# 创建图形和轴，并设置较高的DPI提高分辨率
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=400)  # DPI设置为200

sns.set(style="whitegrid")

# 绘制AUC的折线图
ax1.plot(dropout_rates, mean_auc, marker='o', color='b', label='AUC', linewidth=2)
ax1.set_xlabel('Dropout', fontsize=14)
ax1.set_ylabel('AUC', fontsize=14, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(0.6, 1.2)

# 在折线图上添加数值标签
for i in range(len(dropout_rates)):
    ax1.annotate(f'{mean_auc[i]:.4f}', xy=(dropout_rates[i], mean_auc[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=10, color='b')

# 设置x轴的标签和位置
ax1.set_xticks(dropout_rates)
ax1.set_xticklabels(dropout_rates, fontsize=12)

# 创建一个共享x轴的第二个y轴，用于绘制AUPR的柱状图
ax2 = ax1.twinx()
bars = ax2.bar(dropout_rates, mean_aupr, alpha=0.6, color='r', label='AUPR', width=0.05)
ax2.set_ylabel('AUPR', fontsize=14, color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(0.7, 1.2)

# 在柱状图上添加数值标签
for bar, value in zip(bars, mean_aupr):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}',
             ha='center', va='bottom', fontsize=10, color='r')

# 设置标题
plt.title('AUC and AUPR vs. Dropout Rate', fontsize=16)
fig.tight_layout()

plt.savefig('D:\Project\Mamba\experiment_data\dropout.jpg')
