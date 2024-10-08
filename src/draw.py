import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#示例数据
A1=np.array([0,1,2,3])

Rounds=np.array([0,1,2])

# Cooperation_rate = np.array([
#     [0.9769,0.9786,0.9772,0.9754],
#     [0.9787,0.9771,0.9739,0.9753],
#     [0.9779,0.9772,0.9772,0.9773]
# ])

Cooperation_rate = np.array([
    [0.9713192459568191, 0.9686856868941003, 0.9678864515165501],
    [0.9692830711322286, 0.9681380236767249, 0.9730164112499228],
    [0.9719509533386477, 0.9684468314533585, 0.9708071888717822],
    [0.9652116825300882, 0.9697809707821667,  0.9660935010639499],
])
# 寻找最大值及其位置
max_value = np.max(Cooperation_rate)
max_index = np.unravel_index(np.argmax(Cooperation_rate), Cooperation_rate.shape)

X,Y=np.meshgrid(Rounds,A1)
fig=plt.figure(dpi=600)
ax=fig.add_subplot(111,projection='3d')

#绘制每条线并填充颜色平面
for i in range(Cooperation_rate.shape[0]):
    ax.plot(Y[i],X[i],Cooperation_rate[i])
    ax.plot(Y[i],X[i],np.zeros_like(Cooperation_rate[i]),color='gray',alpha=0.5)#连接到基准平面
#在线条与基准平面之间创建有颜色的平面
    for j in range(len(Rounds)-1):
        verts =[
            [Y[i,j],X[i,j],0.962],
            [Y[i,j+1],X[i,j+1],0.962],
            [Y[i,j+1],X[i,j+1],Cooperation_rate[i,j+1]],
            [Y[i,j],X[i,j],Cooperation_rate[i,j]]
        ]
        # ax.add_collection3d(Poly3DCollection([verts],color=plt.cm.viridis(i/len(A1)),alpha=0.3))
        ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.plasma(i / len(A1)), alpha=0.3))
#注释点的数值，并稍微偏移以避免重叠
    for j in range(Cooperation_rate.shape[1]):
        color = 'red' if (i, j) == max_index else 'black'  # 将最大值标红
        ax.text(Y[i,j],X[i,j],Cooperation_rate[i,j],f'{Cooperation_rate[i,j]:.4f}',color=color,ha='center')
#用灰色虚线连接不同线条之间的对应点
for j in range(Cooperation_rate.shape[1]):
    ax.plot(Y[:,j],X[:,j],Cooperation_rate[:,j],'--',color='gray')
#添加标签并自定义图形
ax.set_zlim(0.962, 0.974)
ax.set_xlabel('d_state')
ax.set_xticklabels([' ','8',' ','16',' ','32',' ','64'])
ax.set_ylabel('expend')
ax.set_yticklabels([' ','1',' ','2',' ','4'])
ax.set_zlabel('  AUPR')
plt.savefig('D:\Project\Mamba\experiment_data\conv_4_AUPR.jpg')