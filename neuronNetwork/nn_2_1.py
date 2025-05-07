from modules import *

# Classifaction
# import matplotlib.pyplot as plt # 可選，用於繪圖可視化

# --- 1. 初始化輸入資料和儲存結果的陣列 ---
# 使用 np.arange 創建輸入資料 x_0 和 x_1
# 範圍是從 -1.0 到 1.0 (不包含 1.0)，步長為 0.1
# 這將產生 20 個點：-1.0, -0.9, ..., 0.9
x_0 = np.arange(-1.0, 1.0, 0.1)
x_1 = np.arange(-1.0, 1.0, 0.1)

# 初始化 Z 陣列，用於儲存 network 的輸出結果
# shape 為 (400, 2)，表示將儲存 400 個樣本的結果，每個結果有 2 個值
# 之所以是 (400, 2)，是因為 x_0 和 x_1 各有 20 個值，總共有 20 * 20 = 400 種輸入組合
# 且 output layer 有 2 個輸出節點 (根據後面的 w_mo 和 b_mo 可以推斷)
Z = np.zeros((400, 2))

# --- 2. 初始化 Weights 和 Biases ---
# 從 input layer 到 hidden layer 的 weights (2 個輸入, 2 個隱藏節點)
w_im = np.array([[1.0, 2.0],
                 [2.0, 3.0]])

# 從 hidden layer 到 output layer 的 weights (2 個隱藏節點, 2 個輸出節點)
# 請注意這裡的 shape 是 (2, 2)，表示 output layer 有 2 個節點
w_mo = np.array([[-1.0, -1.0],
                 [1.0, -1.0]])

# hidden layer 的 biases (2 個隱藏節點)
b_im = np.array([0.3, -0.3])

# output layer 的 biases (2 個輸出節點)
b_mo = np.array([0.4, 0.1])

# --- 3. 定義每一層的計算函數 ---

def middle_layer(x, w, b):
  """
  計算 hidden layer 的輸出。
  輸入: x (來自前一層的輸出), w (當前層的 weights), b (當前層的 biases)
  輸出: 應用 sigmoid activation function 後的結果
  """
  # 計算 weighted sum 加上 bias
  u = np.dot(x, w) + b
  # 應用 sigmoid activation function
  return 1/(1+np.exp(-u))

def output_layer(x, w, b):
  """
  計算 output layer 的輸出。
  輸入: x (來自前一層的輸出), w (當前層的 weights), b (當前層的 biases)
  輸出: 應用 softmax activation function 後的結果 (用於分類)
  """
  # 計算 weighted sum 加上 bias
  u = np.dot(x, w) + b
  # 應用 softmax activation function
  # Softmax 將輸出值轉換為機率分佈，所有輸出值的總和為 1
  return np.exp(u)/np.sum(np.exp(u)) # 這裡的 np.sum(np.exp(u)) 需要注意維度，對於多個樣本，應該在正確的軸上求和

# --- 4. 使用 Nested Loops 計算所有輸入組合的輸出 ---

# 這裡使用 nested loops 來遍歷 x_0 和 x_1 的所有組合
for i in range(20):
  for j in range(20):
    # 從 x_0 和 x_1 中取出第 i 和第 j 個元素，組成一個輸入陣列
    # 注意這裡使用了雙層中括號 [[...]]，這會創建一個 shape 為 (1, 2) 的 NumPy 陣列
    inp = np.array([[x_0[i], x_1[j]]])

    # 呼叫 middle_layer 函數計算 hidden layer 的輸出
    mid = middle_layer(inp, w_im, b_im)

    # 呼叫 output_layer 函數計算 output layer 的輸出 (分類機率)
    out = output_layer(mid, w_mo, b_mo)

    # 將 output layer 的結果 out 存儲到 Z 陣列中
    # i * 20 + j 是一個將二維的 (i, j) 索引轉換為一維陣列索引的方法
    # out 的 shape 是 (1, 2)，所以 Z[i*20 + j] 會存儲這 2 個輸出值
    Z[i*20 + j] = out # if 3 for loops Z[i * 20 + j * 20 + j] = out[0]

# 迴圈結束後，打印出整個 Z 陣列
print(Z)

# --- 可選: 可視化結果 (需要引入 matplotlib) ---
# 這部分的程式碼在您的截圖中沒有，但通常在分類問題中會將 Z 陣列的結果可視化
# 例如，根據輸出的機率值來判斷類別，並用不同的顏色標記在二維平面上
# 如果需要可視化的程式碼，請告訴我。