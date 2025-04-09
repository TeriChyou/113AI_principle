from modules import *
from spicy import stats
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# For future value predicting and R value
from sklearn.metrics import r2_score

# 亂數線性回歸
def basicLinreg():
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.randn(50) 

    slp, ict, r, p, std_err = stats.linregress(x,y)

    def linregFunc(x):
        return slp * x + ict

    model = list(map(linregFunc, x))
    plt.scatter(x, y)
    plt.plot(x, model)
    plt.show()
    print(r)

# Uniform
def uniformDistribution():
    x = np.random.uniform(0.0, 5.0, 100000)
    plt.hist(x)
    plt.show()

# 常態分佈 Normal(Gauss) Distrubution
def normalDistribution():
    x = np.random.normal(0.0, 5.0, 100000)
    plt.hist(x)
    plt.show()

# 基礎linearReg模型的實現y = ax + b 找出 MSE(Mean-Square Error) = 1/n  ​i=1 ∑_n ​(yi ​ − y'​i)^2
def linregFindMSE():
    # 假資料：x 是輸入，y 是有加上雜訊的輸出
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.randn(100) * 2  # y = 2x + 1 加上雜訊
    # 初始化 w 和 b
    w = 0.0
    b = 0.0

    # 學習率
    lr = 0.01

    # 訓練次數
    epochs = 1000
    n = len(x)
    for i in range(epochs):
        y_pred = w * x + b
        error = y - y_pred

        # 計算梯度（對 MSE 的偏微分）
        dw = (-2 / n) * np.sum(x * error)
        db = (-2 / n) * np.sum(error)

        # 更新 w 和 b
        w -= lr * dw
        b -= lr * db

        # 每 100 次印出 loss 看學習情況
        if i % 100 == 0:
            mse = np.mean(error ** 2)
            print(f"Epoch {i}: MSE = {mse:.4f}, w = {w:.4f}, b = {b:.4f}")

    plt.scatter(x, y, label='Data')
    plt.plot(x, w * x + b, color='red', label=f'Fitted line: y = {w:.2f}x + {b:.2f}')
    plt.legend()
    plt.title("Linear Regression from Scratch")
    plt.show()

# 基礎Polynomial model y = w1x^n + w2x^(n-1) + ... + w(n-1)x + b
def polyreg():
    # ======= Step 1: 產生非線性資料 =======
    np.random.seed(42)
    x = np.linspace(-3, 3, 30).reshape(-1, 1)  # reshape 成 (30, 1)
    y = 0.5 * x**3 - x**2 + x + np.random.randn(*x.shape) * 2
    y = y.ravel()  # 拉平 y 變成 (30,)

    # ======= Step 2: 建立多項式特徵轉換器（例如 3 次）=======
    degree = 2
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)  # x_poly 會包含 [1, x, x², x³ ~ x^n]

    # ======= Step 3: 用線性回歸模型來擬合轉換後的特徵 =======
    model = LinearRegression()
    model.fit(x_poly, y)

    # ======= Step 4: 產生預測用 x，包括內插與外插範圍 =======
    x_test = np.linspace(-5, 5, 200).reshape(-1, 1)  # -3~3 是訓練資料的範圍
    x_test_poly = poly.transform(x_test)
    y_pred = model.predict(x_test_poly)

    # ======= Step 5: 畫圖結果，標示出內插與外插區間 =======
    plt.scatter(x, y, label="Training Data")
    plt.plot(x_test, y_pred, color="red", label=f"{degree}-Degree Polynomial Fit")

    # 標出內插範圍區間（x 原始範圍）
    plt.axvline(x.min(), color='gray', linestyle='--', label="Interpolation Region")
    plt.axvline(x.max(), color='gray', linestyle='--')

    plt.title("Polynomial Regression with Interpolation vs. Extrapolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

# 基礎的R值計算
def rValueCalc():
    x = [59, 16, 13, 89, 46, 71, 45, 16, 90, 6, 36, 86, 85, 14, 55, 86, 61, 42, 58, 74]
    y = [86, 63, 31, 33, 8, 36, 43, 44, 12, 80, 84, 23, 85, 60, 45, 72, 61, 15, 54, 28]

    model = np.poly1d(np.polyfit(x,y, 8)) # last num stands for 項次
    modelLine = np.linspace(2, 95, 100)

    plt.scatter(x,y,label=f"{r2_score(y, model(x))}") # R score represents x's fitness of y from F(x)  
    plt.plot(modelLine, model(modelLine))
    plt.show()
    print(r2_score(y, model(x)))

# 基礎的多變數線性模型 y = w1x1 + w2x2 + ... + wnxn + b (預設十維)
def multiCoefLinreg():
    # ======= Step 1. 產生模擬資料 =======
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # X 是 (200, 10) 的資料矩陣
    X = np.random.randn(n_samples, n_features)

    # 隨機產生真實權重（真實的 w）
    true_w = np.random.uniform(-3, 3, size=(n_features,))
    true_b = 5

    # 計算 y，並加入一點 noise
    y = X @ true_w + true_b + np.random.randn(n_samples) * 0.5

    # ======= Step 2. 用公式解出最佳 w 和 b（Normal Equation）=======
    # 增加一列 1 當作 bias 項目
    X_bias = np.hstack((X, np.ones((n_samples, 1))))  # (200, 11)

    # 解 (X^T X)^-1 X^T y
    w_full = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    # 拆出 w 和 b
    w_learned = w_full[:-1]
    b_learned = w_full[-1]

    print("True w:     ", np.round(true_w, 2))
    print("Learned w:  ", np.round(w_learned, 2))
    print("True b:     ", round(true_b, 2))
    print("Learned b:  ", round(b_learned, 2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 只取前兩個特徵做視覺化
    x1 = X[:, 0]
    x2 = X[:, 1]
    y_pred = X @ w_learned + b_learned

    ax.scatter(x1, x2, y, label="True Y", alpha=0.6)
    ax.scatter(x1, x2, y_pred, label="Predicted Y", color='red', s=10)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    ax.set_title("Multiple Linear Regression: 2 Features Visualized")
    ax.legend()
    plt.show()

rValueCalc()