import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
import time


def train_graphsage_with_seeds(data, seed_nodes, num_epochs=5, learning_rate=0.001, device=None):
    # 设备选择
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义 GraphSAGE 模型
    model = GraphSAGE(
        in_channels=32,  # 输入特征维度
        hidden_channels=64,  # 隐藏层维度
        out_channels=4,  # 输出类别数（4 类）
        num_layers=2  # 2 层 GNN
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    model.train()  # 设置为训练模式
    
    log_messages = ""  # 用于累积每轮的log信息

    # 循环每个种子节点
    for seed_node in seed_nodes:
        # 运行 NeighborLoader，使用当前种子节点
        loader = NeighborLoader(
            data,
            input_nodes=torch.tensor([seed_node]),  # 当前种子节点
            num_neighbors=[3, 2],  # 第一层 3 个邻居，第二层 2 个邻居
            batch_size=1,
            replace=False,
            shuffle=False,
        )

        for epoch in range(num_epochs):
            total_loss = 0

            for batch in loader:
                optimizer.zero_grad()
                batch = batch.to(device)  # 迁移到 GPU/CPU

                # 前向传播
                out = model(batch.x, batch.edge_index)

                # 仅计算种子节点的损失
                y_pred = out[:batch.batch_size]  # 仅种子节点的预测
                y_true = batch.y[:batch.batch_size]  # 仅种子节点的真实标签

                loss = F.cross_entropy(y_pred, y_true)  # 交叉熵损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                total_loss += loss.item()

            # 构建本轮的log信息，并将其累积
            log_message = f"Seed Node {seed_node} - Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}\n"
            log_messages += log_message  # 将当前log信息添加到累积字符串中

    return log_messages  # 返回所有的log信息




# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 原始节点特征和标签
x = torch.randn(18, 32)  # 18个节点，每个节点32个特征
y = torch.randint(0, 4, (18, ))  # 18个节点，每个节点一个标签，标签范围[0, 3]

# 原始边 (0-7节点之间的连接关系)
edge_index = torch.tensor([
    [2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    [0, 0, 1, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)


# 假设已有数据集 data，种子节点为 [0, 1, 2]
# 训练
seed_nodes = [0, 1]
log_output = train_graphsage_with_seeds(data, seed_nodes)
print(log_output)  # 打印所有训练过程的log信息




# 函数：用于获取某个种子节点的邻居
def get_neighbors(seed_node):
    loader = NeighborLoader(
        data,
        input_nodes=torch.tensor([seed_node]),  # 当前种子节点
        num_neighbors=[3, 2],  # 第一层 3 个邻居，第二层 2 个邻居
        batch_size=1,
        replace=False,
        shuffle=False,
    )

    # 获取采样结果
    batch = next(iter(loader))
    nodes_visited = batch.n_id.tolist()  # 采样到的所有节点
    edges_visited = batch.edge_index.t().tolist()  # 采样到的边

    # 获取第一层和第二层邻居
    first_layer_neighbors = nodes_visited[1:4]  # 第一层邻居（从第二个到第四个节点）
    second_layer_neighbors = nodes_visited[4:6]  # 第二层邻居（从第五个到第六个节点）

    return first_layer_neighbors, second_layer_neighbors

# 获取节点0和节点1的邻居
first_layer_neighbors_0, second_layer_neighbors_0 = get_neighbors(0)
first_layer_neighbors_1, second_layer_neighbors_1 = get_neighbors(1)

# 采样步骤：先从节点0开始，再从节点1开始
sampling_steps = [
    {"highlight_nodes": [], "highlight_edges": []},  # 初始状态，只有种子节点
    {"highlight_nodes": first_layer_neighbors_0, "highlight_edges": [(0, n) for n in first_layer_neighbors_0]},  # 1-hop邻居
    {"highlight_nodes": second_layer_neighbors_0, "highlight_edges": [(n, nn) for n, nn in zip(first_layer_neighbors_0, second_layer_neighbors_0)]},  # 2-hop邻居
    {"highlight_nodes": [1], "highlight_edges": []},  # 节点1为新的种子节点
    {"highlight_nodes": first_layer_neighbors_1, "highlight_edges": [(1, n) for n in first_layer_neighbors_1]},  # 1-hop邻居
    {"highlight_nodes": second_layer_neighbors_1, "highlight_edges": [(n, nn) for n, nn in zip(first_layer_neighbors_1, second_layer_neighbors_1)]},  # 2-hop邻居
]

# 转换为 NetworkX 图
G = nx.Graph()
G.add_edges_from(edge_index.t().tolist())  # 原始图

# 获取坐标布局并确保节点 0 在中心
pos = nx.spring_layout(G, seed=42, center=[0, 0], k=0.5)  # 让 0 号节点居中

# 初始化绘图
fig, ax = plt.subplots(figsize=(6, 4))

# 更新函数：根据步骤动态绘制图形
def update(num):
    
    # 1. 复位颜色
    current_node_colors = {node: "lightgray" for node in G.nodes}
    current_edge_colors = {tuple(edge): "lightgray" for edge in G.edges}

    # 2. 根据步骤设置颜色
    step = sampling_steps[num]

    # 保持当前种子节点的颜色为红色
    if num == 0 or num == 3:
        current_node_colors[0 if num == 0 else 1] = "red"
  
    if num < 3:
        current_node_colors[0] = "red"
        ax.cla()  # 避免画面固定
        ax.set_title(f"Seed0: Neighbor Sampling Process")

    if num == 2:
        for n in first_layer_neighbors_0:
            current_node_colors[n] = "orange"
    

    # 3. 更新采样的节点和边颜色
    for n in step["highlight_nodes"]:
        current_node_colors[n] = "orange" if num == 1 or num == 4 else "yellow"  # 1-hop 邻居橙色，2-hop 邻居黄色
    for e in step["highlight_edges"]:
        current_edge_colors[tuple(e)] = "orange" if num == 1 or num == 4 else "yellow"
    
    if num >= 3:
        current_node_colors[1] = "red"
        ax.cla()  # 避免画面固定
        ax.set_title(f"Seed1: Neighbor Sampling Process")
    if num >= 4:
        for n in first_layer_neighbors_1:
            current_node_colors[n] = "orange"

    # 4. 绘制网络图
    nx.draw_networkx_nodes(G, pos, node_color=[current_node_colors[n] for n in G.nodes], ax=ax, node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color=[current_edge_colors.get((u, v), "lightgray") for u, v in G.edges], ax=ax, width=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

    # 5. 在右下角显示一串数字
    #if num == 5:
    #    seed_nodes = [0, 1]
    #    log_output = train_graphsage_with_seeds(data, seed_nodes)
    #    ax.text(0.95, 0.05, log_output, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=12, color='blue')


# 创建动画（循环播放）
ani = animation.FuncAnimation(fig, update, frames=len(sampling_steps), interval=2000, repeat=True)

plt.show()
