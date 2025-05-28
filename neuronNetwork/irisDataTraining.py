from modules import *

# import the data # 花萼長度、寬度、花瓣長度、品種名稱 150筆資料
df = pd.read_csv('irisdata/iris.data', header=None)

# 資料預處理 取前百 僅區別兩種 分成訓練集40 & 測試集10

x = df.iloc[0:100, [0,1,2,3]].values
y = df.iloc[0:100, 4].values

y = np.where(y == 'iris-setosa', 0, 1)

x_train = np.empty((80, 4)) # 訓練資料有八十筆
x_test = np.empty((20, 4)) # 測試資料有二十筆
y_train = np.empty(80) # 訓練的答案有八十筆
y_test = np.empty(20) # 要測試的答案有二十筆
x_train[:40], x_train[40:] = x[:40], x[50:90] 
x_test[:10], x_test[10:] = x[40:50], x[90:100]
y_train[:40], y_train[40:] = y[:40], y[50:90]
y_test[:10], y_test[10:] = y[40:50], y[90:100]

print(x_train)
print(y_train)

# 函數區

def sigmoid(x, w, b):
    # Sigmoid 函數，將線性組合結果轉換為機率值
    # x: 特徵資料 (樣本數, 特徵數)
    # w: 權重 (特徵數,)
    # b: 偏差 (單一值)
    u = np.dot(x, w) + b  # 線性組合
    return 1/(1+np.exp(-u))  # Sigmoid 轉換

def update(x, y, w, b, eta):
    # 權重與偏差更新函數
    # x: 訓練資料 (樣本數, 特徵數)
    # y: 標籤 (樣本數,)
    # w: 權重 (特徵數,)
    # b: 偏差 (單一值)
    # eta: 學習率 (單一值) 越小收斂越慢
    y_pred = sigmoid(x, w, b)  # 預測值
    a = (y_pred - y_train) * y_pred * (1 - y_pred)  # 誤差修正量 1-y_pred 為 預測為0的機率
    for i in range(4):
        w[i] -= eta * 1/float(len(y)) * np.sum(a*x[:,i])  # 更新每個特徵的權重 len(y) 為訓練資料的樣本數
    b -= eta*1/float(len(y)) * np.sum(a)  # 更新偏差
    return w, b

# Main training stuffs
weights = np.ones(4)/10
bias = np.ones(1)/10
eta = 0.1

for _ in range(10): # epochs
    weights, bias = update(x_train, y_train, weights, bias, eta)

print('weights=',weights,'bias=', bias)
print(sigmoid(x_test, weights, bias))