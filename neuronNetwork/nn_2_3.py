from modules import *

x_0 = np.arange(-1.0, 1.0, 0.1)
x_1 = np.arange(-1.0, 1.0, 0.1)
Z = np.zeros((400, 3)) # n = 3
# weight of iptLayer to hidLayer 輸入節點數量*隱藏節點數量 3,3 
w_im = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 2.5]
])
# weight of hidLayer to optLayer 隱藏節點數量*輸出節點數量 3,3
w_mo = np.array([
    [-1.0, 1.0, 1.0],
    [1.0, -1.0, 2.0],
    [1.5, -1.5, 3.0]
])
# bias of hidLayer 隱藏節點數量
b_im = np.array([0.3, -0.3, 1.0])
# bias of optLayer 輸出節點數量
b_mo = np.array([0.4, 0.1, 2.5])

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

for i in range(20):
  for j in range(20):
    inp = np.array([x_0[i], x_1[j]])
    mid = middle_layer(inp, w_im, b_im)
    out = output_layer(mid, w_mo, b_mo)
    Z[i*20 + j] = out

print(Z) # 回傳的400筆資料當中 每個陣列裡面n個值加起來都是1 因為是機率分佈

# TASK 改三個神經元(hidden layer) output to 3