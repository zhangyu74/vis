import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# 采样 100 个潜变量点 (标准二维正态分布)
n_samples = 200
z1 = np.random.normal(0, 1, n_samples)
z2 = np.random.normal(0, 1, n_samples)

# 定义非线性映射 g(z)
x1 = np.sin(z1) + 0.2 * z2
x2 = np.cos(z1) + 0.2 * z2

# 创建动画，调整子图为上下排列
fig, ax = plt.subplots(2, 1, figsize=(6, 10))

# 上图：潜变量分布
ax[0].set_title("Latent Variable Space                    2D Normal (z)")
ax[0].set_xlabel("z1")
ax[0].set_ylabel("z2")

# 下图：变换后数据分布
ax[1].set_title("  Data Space                             Moon Shape (x=g(z))")
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")

# 绘制 KDE 密度图
sns.kdeplot(x=z1, y=z2, fill=True, cmap="Blues", alpha=0.5, ax=ax[0])
sns.kdeplot(x=x1, y=x2, fill=True, cmap="Reds", alpha=0.5, ax=ax[1])

# 绘制曲线
z3 = np.linspace(-3, 3, 100)  
z4 = np.linspace(-3, 3, 100)  
x3 = np.sin(z3) + 0.2 * z4
x4 = np.cos(z3) + 0.2 * z4
ax[1].plot(x3, x4, color='green', alpha=0.5, label='g(z)')
ax[1].legend()

# 初始化散点图
scatter_top = ax[0].scatter([], [], alpha=0.5, color='blue')
scatter_bottom = ax[1].scatter([], [], alpha=0.5, color='red')

def init():
    scatter_top.set_offsets(np.empty((0, 2)))
    scatter_bottom.set_offsets(np.empty((0, 2)))
    return scatter_top, scatter_bottom

def update(frame):
    # 上图散点图逐渐减少
    scatter_top.set_offsets(np.column_stack((z1[:n_samples-frame], z2[:n_samples-frame])))
    
    # 下图散点图逐渐增加
    scatter_bottom.set_offsets(np.column_stack((x1[:frame], x2[:frame])))
    
    return scatter_top, scatter_bottom

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=n_samples, init_func=init, interval=50, blit=True)
plt.tight_layout()  # 调整布局，防止标题重叠
plt.show()
