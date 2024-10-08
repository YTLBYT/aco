import numpy as np


def getRoadNetwork():
    # 定义16个节点的邻接矩阵
    inf = float('inf')
    road_network = np.array([
        [0, 2, inf, inf, 1, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
        [2, 0, 3, inf, inf, 1, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
        [inf, 3, 0, 4, inf, inf, 2, inf, inf, inf, inf, inf, inf, inf, inf, inf],
        [inf, inf, 4, 0, inf, inf, inf, 5, inf, inf, inf, inf, inf, inf, inf, inf],
        [1, inf, inf, inf, 0, 2, inf, inf, 2, inf, inf, inf, inf, inf, inf, inf],
        [inf, 1, inf, inf, 2, 0, 3, inf, inf, 2, inf, inf, inf, inf, inf, inf],
        [inf, inf, 2, inf, inf, 3, 0, 4, inf, inf, 1, inf, inf, inf, inf, inf],
        [inf, inf, inf, 5, inf, inf, 4, 0, inf, inf, inf, 2, inf, inf, inf, inf],
        [inf, inf, inf, inf, 2, inf, inf, inf, 0, 3, inf, inf, 2, inf, inf, inf],
        [inf, inf, inf, inf, inf, 2, inf, inf, 3, 0, 2, inf, inf, 1, inf, inf],
        [inf, inf, inf, inf, inf, inf, 1, inf, inf, 2, 0, 1, inf, inf, 2, inf],
        [inf, inf, inf, inf, inf, inf, inf, 2, inf, inf, 1, 0, inf, inf, inf, 3],
        [inf, inf, inf, inf, inf, inf, inf, inf, 2, inf, inf, inf, 0, 2, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf, inf, 1, inf, inf, 2, 0, 3, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2, inf, inf, 3, 0, 2],
        [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3, inf, inf, 2, 0]
    ])
    return road_network
