from modules import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


# ====== STEP 0: 全域變數（給之後決定用）======
y_col_index = 1  # 先預設使用 y1（即第二欄），可以改成 2 or 3 看學號
max_degree = 10
noise_scale = 1.0
csv_file = 'your_file.csv'  # ←請換成實際檔案名稱

# ====== STEP 1: 讀取資料並轉 numpy ======
df = pd.read_csv(csv_file, usecols=[0, y_col_index])  # 只選 x 與 y1/2/3
df_arr = df.to_numpy()
x_raw = df_arr[:, 0].reshape(-1, 1)
y_clean = df_arr[:, 1]

# ====== STEP 2: 加入隨機雜訊 ======
# 隨便選三種不同雜訊
noise = (
    np.array([np.random.uniform(-1, 1) for _ in range(len(x_raw))]) +
    np.random.normal(0, noise_scale, size=len(x_raw)) +
    np.random.logistic(0, noise_scale, size=len(x_raw))
)
y_noisy = y_clean + noise

# ====== STEP 3: 找最佳多項式次數（誤差差距 < 1%）======
best_degree = 1
prev_mse = float('inf')

for degree in range(1, max_degree + 1):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x_raw)
    
    model = LinearRegression().fit(x_poly, y_noisy)
    y_pred = model.predict(x_poly)
    mse = mean_squared_error(y_noisy, y_pred)

    if abs(prev_mse - mse) / prev_mse < 0.01:
        best_degree = degree - 1  # 上一個就夠好
        break
    prev_mse = mse

# ====== STEP 4: 視覺化最佳 fit 曲線 ======
poly_best = PolynomialFeatures(best_degree)
x_poly_best = poly_best.fit_transform(x_raw)
y_fit = LinearRegression().fit(x_poly_best, y_noisy).predict(x_poly_best)

plt.scatter(x_raw, y_noisy, label="Noisy Data")
plt.plot(x_raw, y_fit, color='red', label=f"Poly Fit (deg={best_degree})")
plt.title("Polynomial Regression Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# ====== STEP 5: 轉為 a × b 矩陣後算行列式與反矩陣 ======
# 假設 reshape 成方陣 a x a（最近接平方數）
length = len(x_raw)
a = int(np.floor(np.sqrt(length)))
b = a

x_matrix = x_raw[:a*b].reshape(a, b)
y_matrix = y_noisy[:a*b].reshape(a, b)

# 任選一個做運算
try:
    det = np.linalg.det(y_matrix)
    inverse = np.linalg.inv(y_matrix)

    print(f"\nDeterminant = {det:.3f}")
    identity_check = y_matrix @ inverse
    print("y_matrix * inverse ≈ identity matrix?\n", np.round(identity_check, 2))
except np.linalg.LinAlgError:
    print("❌ 此矩陣無反矩陣（可能是奇異矩陣）")
