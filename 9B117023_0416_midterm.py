import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Set random seed :)
np.random.seed(20250416)


# 1.讀取資料（用 y3，依學號決定 9B117023最後一碼3%3=0 作1c, 2c)
df = pd.read_csv("midterm_113_2.csv", usecols=[0, 3])  # x, y3
data = df.to_numpy()
x = data[:, 0]
y_clean = data[:, 1]
"""
確認用而已
for i in range(len(x)):
    print(x[i], y_clean[i])
"""

# 2.加入Uniform distribution雜訊 & 印出 Y的Mean value 和 StdDeviation
# noise_type = "uniform" # 1c 要求 "uniform"
noise = np.random.uniform(-1, 1, size=len(x))

y = y_clean + noise
print(f"1-1 Y's mean value:{np.mean(y)}")
print(f"1-2 Y's standard deviation:{np.std(y)}")

# 3. 三次多項式線性回歸
curr_d = 3
# Final model
model = np.poly1d(np.polyfit(x, y, curr_d))
r2Score = f"{r2_score(y, model(x))}"
print(f"2-1 相關係數R²={r2Score}")
print("2-2&2-3 請看跳出來的圖(關閉後繼續顯示接下來題目的答案):")
# Plot drawing and display R² score
x_line = np.linspace(min(x), max(x), 100)
plt.scatter(x, y, label=f"R²: {r2Score}")
plt.plot(x_line, model(x_line), color='red', label=f"Poly deg={curr_d}")
plt.gcf().canvas.manager.set_window_title("2-2 & 2-3 Plot")
plt.title("Y vs X by 9B117023")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.grid()
plt.show()


# 4. reshape → Calculate for determination and rev-matrix =====
length = len(x)
a = int(np.floor(np.sqrt(length)))
b = a

y_mat = y[:a*b].reshape(a, b)
print(f"4. Reshaped Y: {np.round(y_mat, 3)}")
# 5. stdNo tail num = 17023
idTail = "17023"

# 建構 5x1 的矩陣 Yb
Yb = np.array([[int(d)] for d in idTail])
print(f"5.學號末五碼矩陣Yb={Yb}")

# 6.(1)~(4)
det = np.linalg.det(y_mat)
print(f"6-1. Y55 det: {np.round(det, 3)}")

try:
    inv = np.linalg.inv(y_mat)
    print(f"6-2. Y55 invMtx: {np.round(inv, 2)}")
except np.linalg.LinAlgError:
    print(" 無法計算反矩陣（可能是奇異矩陣）")

sum_axis_0 = np.sum(y_mat, axis=0)
print("6-3. 沿 axis=0 的加總（每一列相加）:\n", sum_axis_0)

sum_axis_1 = np.sum(y_mat, axis=1)
print("6-4. 沿 axis=1 的加總（每一行相加）:\n", sum_axis_1)

# 7.(1)~(3)
print(f"7-1 Y55．Yb 矩陣內積={np.round(np.dot(y_mat, Yb),2)}, \n")

try:
    inv = np.linalg.inv(y_mat)
    dotVal = np.dot(inv, Yb)
    print("7-2 Yinv ⋅ Yb =\n", np.round(dotVal, 2))
except np.linalg.LinAlgError:
    print("7-2 Y55 無法反轉，無法做 dot(Yinv, Yb)")


try:
    inv = np.linalg.inv(y_mat)
    identity = np.dot(y_mat, inv)
    print("7-3 Y55 ⋅ Yinv =\n", np.round(identity, 2))
    print("是不是單位矩陣？", np.allclose(identity, np.eye(5)))
except:
    print("7-3 無法驗證 Y55 ⋅ Yinv 是否為單位矩陣")

# Bonus
# Find best degree (R² 差 < 0.1%)
print("\n原本上週聽說要找best degree 前後者差距小於1%的項次，\n但這次沒有還是想做一下：")

max_degree = 10
best_degree = 1
prev_r2 = -np.inf

for d in range(1, max_degree + 1):
    model = np.poly1d(np.polyfit(x, y, d))
    r2 = r2_score(y, model(x))
    print(f"Degree:{d}, R²:{r2}")
    if d > 1 and abs(r2 - prev_r2) < 0.001:
        best_degree = d - 1
        print(f"差距小於0.1% 因此best degree為:{best_degree}")
        break
    prev_r2 = r2

# Final model
model = np.poly1d(np.polyfit(x, y, best_degree))
# plot out
x_line = np.linspace(min(x), max(x), 100)
plt.scatter(x, y, label=f"R²: {r2_score(y, model(x)):.10f}")
plt.plot(x_line, model(x_line), color='red', label=f"Poly deg={best_degree}")
plt.gcf().canvas.manager.set_window_title("Terry's Bonus Presentation - 9B117023")
plt.title("Find best R² value")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.grid()
plt.show()
