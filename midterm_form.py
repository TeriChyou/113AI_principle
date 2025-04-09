import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random

# ===== STEP 1: 讀取資料（假設用 y1，依學號決定）=====
df = pd.read_csv("your_file.csv", usecols=[0, 1])  # x, y1
data = df.to_numpy()
x = data[:, 0]
y_clean = data[:, 1]

# ===== STEP 2: 加入一種隨機雜訊（選一種） =====
noise_type = "uniform" # "uniform", "normal", "logistic"
if noise_type == "uniform":
    noise = np.random.uniform(-1, 1, size=len(x))
elif noise_type == "normal":
    noise = np.random.normal(0, 1, size=len(x))
else:
    noise = np.random.logistic(0, 1, size=len(x))

y = y_clean + noise

# ===== STEP 3: 自動尋找最佳 degree（R² 差 < 1%）=====
max_degree = 10
best_degree = 1
prev_r2 = -np.inf

for d in range(1, max_degree + 1):
    model = np.poly1d(np.polyfit(x, y, d))
    r2 = r2_score(y, model(x))
    if d > 1 and abs(r2 - prev_r2) < 0.01:
        best_degree = d - 1
        break
    prev_r2 = r2

# 最終模型
model = np.poly1d(np.polyfit(x, y, best_degree))

# ===== STEP 4: 畫圖 + 顯示 R² 分數 =====
x_line = np.linspace(min(x), max(x), 100)
plt.scatter(x, y, label=f"R²: {r2_score(y, model(x)):.4f}")
plt.plot(x_line, model(x_line), color='red', label=f"Poly deg={best_degree}")
plt.title("Polynomial Regression (最佳次數)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# ===== STEP 5: reshape → 計算行列式與反矩陣驗證 =====
length = len(x)
a = int(np.floor(np.sqrt(length)))
b = a

y_mat = y[:a*b].reshape(a, b)

try:
    det = np.linalg.det(y_mat)
    inv = np.linalg.inv(y_mat)
    identity = y_mat @ inv # np.dot(y_mat, inv)
    print(f"行列式 (det): {det:.2f}")
    print("反矩陣乘原矩陣 ≈ 單位矩陣？")
    print(np.round(identity, 2))
except np.linalg.LinAlgError:
    print("❌ 無法計算反矩陣（可能是奇異矩陣）")
