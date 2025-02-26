from modules import *

# 創建矩陣
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩陣加法
C = A + B

# 矩陣乘法（點乘）
D = np.dot(A, B)  # or A @ B

# 轉置矩陣
E = A.T

# 逆矩陣
F = np.linalg.inv(A)

# 特徵值與特徵向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print(D)
print(E)
print(F)
print(f"{eigenvalues}, {eigenvectors}")
