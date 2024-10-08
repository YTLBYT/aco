import numpy as np
import random
import matplotlib.pyplot as plt
from road_network import getRoadNetwork


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
        根据蚁群算法生成一条路径
        start_node: 起点
        excluded_nodes: 不能经过的节点集合
        end_node: 终点
        返回值: 生成的路径
        """
        route = [start_node]  # 初始化路径，包含起点
        visited = set([start_node])  # 初始化已访问节点集合
        while len(visited) < len(self.distances) - len(excluded_nodes):
            # 选择下一个节点
            next_node = self._select_next_node(self.pheromone[route[-1]], self.distances[route[-1]], visited)
            if next_node == end_node:  # 如果到达终点，则停止
                break
            if next_node in excluded_nodes:  # 如果下一个节点在排除列表中，则跳过
                continue
            route.append(next_node)  # 将下一个节点加入路径
            visited.add(next_node)  # 将该节点标记为已访问
        route.append(end_node)  # 将终点加入路径
        return route

    def _update_pheromone(self, all_routes, best_route):
        """
        更新信息素矩阵，基于所有路径和最佳路径
        all_routes: 所有蚂蚁走过的路径
        best_route: 迭代中发现的最佳路径
        """
        self.pheromone *= self.decay  # 信息素衰减
        for route in all_routes:
            for i in range(len(route) - 1):
                # 更新信息素，距离越短的路径贡献的越多
                self.pheromone[route[i]][route[i + 1]] += 1.0 / self._route_distance(route)

    def run(self, start_node, end_node, excluded_nodes):
        """
        运行蚁群算法，找到最优路径
        start_node: 起点
        end_node: 终点
        excluded_nodes: 不能经过的节点集合
        返回值: 最佳路径
        """
        best_route = None  # 记录最佳路径
        shortest_distance = float('inf')  # 记录最短距离
        for iteration in range(self.n_iterations):
            all_routes = []  # 保存所有蚂蚁生成的路径
            for _ in range(self.n_ants):
                # 生成每只蚂蚁的路径
                route = self._generate_route(start_node, excluded_nodes, end_node)
                all_routes.append(route)
                distance = self._route_distance(route)  # 计算路径的距离
                if distance < shortest_distance:  # 如果找到更短的路径，则更新最佳路径
                    best_route = route
                    shortest_distance = distance
            self._update_pheromone(all_routes, best_route)  # 更新信息素矩阵
        return best_route


# 路网绘制函数
def plot_route(matrix, route, excluded_nodes):
    """
    绘制路网和最佳路径
    matrix: 邻接矩阵
    route: 最佳路径
    excluded_nodes: 排除的节点集合
    """
    fig, ax = plt.subplots()
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if matrix[i][j] != float('inf'):
                # 用灰色线条表示所有连接
                ax.plot([i, j], [i, j], 'grey')

    # 用红色线条绘制最优路径
    for i in range(len(route) - 1):
        ax.plot([route[i], route[i + 1]], [route[i], route[i + 1]], 'red')

    ax.scatter(range(len(matrix)), range(len(matrix)), c='blue')  # 用蓝色点绘制节点
    ax.scatter(excluded_nodes, excluded_nodes, c='black', label='Excluded Nodes')  # 用黑色点绘制不可经过的节点
    ax.set_title("Optimal Path")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 获取路网
    matrix = np.array(getRoadNetwork())

    # 蚁群算法参数
    n_ants = 10  # 蚂蚁数量
    n_iterations = 10  # 最大迭代次数
    decay = 0.95  # 信息素衰减系数
    start_node = 0  # 起点
    end_node = 15  # 终点
    excluded_nodes = [2, 7]  # 排除的节点

    # 运行蚁群算法
    ant_colony = AntColony(matrix, n_ants, n_iterations, decay, alpha=1, beta=2)
    best_route = ant_colony.run(start_node, end_node, excluded_nodes)

    # 打印路径节点
    print("最佳路径:", best_route)

    # 绘制路径
    plot_route(matrix, best_route, excluded_nodes)
