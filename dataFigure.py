import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import road_network

matplotlib.use('TkAgg')
inf = float('inf')

road_network = road_network.getRoadNetwork()

# 创建图
G = nx.Graph()

# 添加节点
for i in range(len(road_network)):
    G.add_node(i)

# 添加边
for i in range(len(road_network)):
    for j in range(i + 1, len(road_network)):
        if road_network[i][j] != inf:
            G.add_edge(i, j, weight=road_network[i][j])

# 布局调整,使用spring布局绘制图
pos = nx.spring_layout(G, seed=42)

# 获取边的权重
labels = nx.get_edge_attributes(G, 'weight')

# 绘制图
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600, font_size=10, font_color='black',
        font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
nx.draw(G, pos, edge_color='gray', width=2, alpha=0.7)

plt.title('City Road Network with 16 Nodes')
plt.axis('equal')  # 保持坐标轴比例相等
plt.show()