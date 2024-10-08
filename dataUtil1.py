import numpy as np

# 给定的矩阵
inf = float('inf')
matrix = [
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
]

# 转换为 NumPy 数组以便于处理
matrix_np = np.array(matrix)

# 检查对称性
n = matrix_np.shape[0]
is_symmetric = True
for i in range(n):
    for j in range(n):
        if matrix_np[i][j] != matrix_np[j][i]:  # 如果不相等
            is_symmetric = False
            print(f"不对称元素: matrix[{i}][{j}] = {matrix_np[i][j]}, matrix[{j}][{i}] = {matrix_np[j][i]}")

if is_symmetric:
    print("该矩阵是对称矩阵。")
else:
    print("该矩阵不是对称矩阵。")
