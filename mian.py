import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib
from road_network import getRoadNetwork

matplotlib.use('TkAgg')


# 蚁群算法类
class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay, alpha=1, beta=1):
        """
        初始化蚁群算法参数：
        distances: 邻接矩阵，表示节点之间的距离
        n_ants: 蚂蚁数量
        n_iterations: 迭代次数
        decay: 信息素衰减系数
        alpha: 信息素重要性权重
        beta: 距离重要性权重
        """
        self.distances = distances  # 距离矩阵
        self.pheromone = np.ones(self.distances.shape) / len(distances)  # 初始化信息素矩阵
        self.all_indices = range(len(distances))  # 全部节点的索引
        self.n_ants = n_ants  # 蚂蚁的数量
        self.n_iterations = n_iterations  # 最大迭代次数
        self.decay = decay  # 信息素衰减参数
        self.alpha = alpha  # 信息素的重要性
        self.beta = beta  # 距离的重要性

    def _calculate_total_distance(self, route):
        """
        计算给定路径的总距离
        """
        total_distance = sum(self.distances[route[i], route[i + 1]] for i in range(len(route) - 1))
        return total_distance

    def _route_distance(self, route):
        """
        计算给定路径的总距离
        route: 蚂蚁选择的路径
        返回值: 路径的总距离
        """
        return sum([self.distances[route[i], route[i + 1]] for i in range(len(route) - 1)])

    def _select_next_node(self, pheromone, distances, visited):
        """
        根据信息素和距离选择下一个节点
        pheromone: 当前节点到其他节点的所有信息素值
        distances: 当前节点到其他节点的距离
        visited: 已访问的节点集合
        返回值: 选择的下一个节点
        """
        pheromone = np.copy(pheromone)  # 复制信息素数组
        pheromone[list(visited)] = 0  # 已访问的节点信息素设置为0，避免重复访问

        # 处理无效距离
        valid_distances = np.copy(distances)
        valid_distances[valid_distances == float('inf')] = 0  # 无穷距离设为0，表示不可访问

        # 确保没有距离为0，除以0将会导致错误
        valid_distances[valid_distances == 0] = np.inf  # 避免除以0

        with np.errstate(divide='ignore', invalid='ignore'):
            row = pheromone ** self.alpha * ((1.0 / valid_distances) ** self.beta)
            row[np.isnan(row)] = 0  # 将 NaN 值设置为 0，确保没有无效概率

        total = row.sum()

        # 如果概率总和为0，说明所有可达节点已经访问过或不可达
        if total == 0:
            return None

        norm_row = row / total  # 归一化，确保概率和为1

        # 使用概率选择下一个节点
        return np.random.choice(self.all_indices, 1, p=norm_row)[0]

    def _generate_route(self, start_node, excluded_nodes, end_node):
        """
        为单只蚂蚁生成一条从 start_node 到 end_node 的路径。
        """
        route = [start_node]
        visited = set(excluded_nodes)  # 标记已访问节点（包括排除的节点）
        visited.add(start_node)  # 将起始节点标记为已访问

        print(f"开始生成路径，起始节点：{start_node}")

        while route[-1] != end_node:
            current_node = route[-1]
            print(f"当前节点: {current_node}, 已访问节点: {visited}")

            next_node = self._select_next_node(self.pheromone[current_node], self.distances[current_node], visited)
            if next_node is None:
                print(f"从节点 {current_node} 无法找到有效的下一个节点，结束路径生成。")
                return None

            route.append(next_node)
            visited.add(next_node)

        print(f"生成的路径：{route}")
        return route

    def _update_pheromone(self, all_routes, best_route):
        """
        更新信息素矩阵，基于所有路径和最佳路径
        all_routes: 所有蚂蚁走过的路径
        best_route: 迭代中发现的最佳路径
        """
        self.pheromone *= self.decay  # 信息素衰减
        for route, _ in all_routes:
            if route is None or len(route) < 2:
                continue  # 跳过无效路径

            for i in range(len(route) - 1):
                node_i = route[i]
                node_next = route[i + 1]

                if node_i < 0 or node_i >= len(self.pheromone) or node_next < 0 or node_next >= len(self.pheromone):
                    print(f"警告: 路径索引无效，跳过 ({node_i}, {node_next})")
                    continue

                # 更新信息素，距离越短的路径贡献的越多
                self.pheromone[node_i][node_next] += 1.0 / self._route_distance(route)

    def run(self, start_node, end_node, excluded_nodes):
        """
        运行蚁群算法，寻找从 start_node 到 end_node 的最优路径。
        """
        best_route = None
        best_distance = float('inf')

        for iteration in range(self.n_iterations):
            print(f"第 {iteration + 1} 次迭代开始")

            all_routes = []
            for ant in range(self.n_ants):
                print(f"\n蚂蚁 {ant + 1} 开始寻找路径")
                route = self._generate_route(start_node, excluded_nodes, end_node)

                if route is not None:
                    distance = self._calculate_total_distance(route)
                    all_routes.append((route, distance))
                    print(f"蚂蚁 {ant + 1} 找到的路径：{route}，路径长度：{distance}")
                else:
                    print(f"蚂蚁 {ant + 1} 未找到有效路径")

            # 找到本轮的最优路径
            if all_routes:
                best_in_iteration = min(all_routes, key=lambda x: x[1])
                print(f"\n第 {iteration + 1} 次迭代最佳路径：{best_in_iteration[0]}，路径长度：{best_in_iteration[1]}")

                if best_in_iteration[1] < best_distance:
                    best_route = best_in_iteration[0]
                    best_distance = best_in_iteration[1]
                    print(f"更新全局最佳路径为：{best_route}，路径长度：{best_distance}")

            # 更新信息素
            self._update_pheromone(all_routes, best_route)
            print(f"第 {iteration + 1} 次迭代后信息素更新完毕\n")

        print(f"\n算法结束，最优路径：{best_route}，最短距离：{best_distance}")
        return best_route


# 路网绘制函数
def plot_route(matrix, best_route, excluded_nodes):
    """
    绘制路网和最佳路径
    matrix: 邻接矩阵
    best_route: 最佳路径
    excluded_nodes: 排除的节点集合
    """
    # 创建图
    G = nx.Graph()

    # 添加节点
    for i in range(len(matrix)):
        G.add_node(i)

    # 添加边
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if matrix[i][j] != float('inf'):  # 确保使用 float('inf')
                G.add_edge(i, j, weight=matrix[i][j])

    # 布局调整, 使用 spring 布局绘制图
    pos = nx.spring_layout(G, seed=42)

    # 获取边的权重
    labels = nx.get_edge_attributes(G, 'weight')

    # 设置节点颜色
    node_color = ['black' if node in excluded_nodes else 'lightblue' for node in G.nodes()]

    # 绘制图
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=600, font_size=10, font_color='black',
            font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='green')
    nx.draw(G, pos, edge_color='gray', width=2, alpha=0.7)

    # 绘制最优路径
    if best_route is not None:
        path_edges = [(best_route[i], best_route[i + 1]) for i in range(len(best_route) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)

    plt.title('City Road Network with Optimal Path')
    plt.axis('equal')  # 保持坐标轴比例相等
    plt.show()


if __name__ == "__main__":
    # 获取路网
    matrix = np.array(getRoadNetwork())

    # 蚁群算法参数
    n_ants = 10  # 蚂蚁数量
    n_iterations = 100  # 最大迭代次数
    decay = 0.95  # 信息素衰减系数
    start_node = 5  # 起点
    end_node = 15  # 终点
    excluded_nodes = [6, 9, 10]  # 排除的节点

    # 运行蚁群算法
    ant_colony = AntColony(matrix, n_ants, n_iterations, decay, alpha=1, beta=2)
    best_route = ant_colony.run(start_node, end_node, excluded_nodes)

    # 绘制路径
    plot_route(matrix, best_route, excluded_nodes)
