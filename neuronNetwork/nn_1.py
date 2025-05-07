from modules import *

# --- 1. 定義 Sigmoid Activation Function ---
def sigmoid(x):
  """
  Sigmoid 激活函數。
  它將輸入值壓縮到 0 到 1 之間。
  """
  return 1 / (1 + np.exp(-x))

# --- 2. 初始化 Weights 和 Biases (來自截圖) ---
# 從 input layer 到 hidden layer 的 weights (2 個輸入, 2 個隱藏節點)
# Shape 是 (輸入數量, 隱藏節點數量)
w_im = np.array([[4.0, 4.0],
                 [4.0, 4.0]])

# hidden layer 的 biases (2 個隱藏節點)
b_im = np.array([3.0, -3.0])

# 從 hidden layer 到 output layer 的 weights (2 個隱藏節點, 1 個輸出節點)
# Shape 是 (隱藏節點數量, 輸出節點數量)
w_mo = np.array([[1.0],
                 [-1.0]])

# output layer 的 bias (1 個輸出節點)
b_mo = np.array([0.1])

# --- 3. 定義 Neural Network Class ---
class ThreeLayerNN:
    def __init__(self, w_im, b_im, w_mo, b_mo):
        """
        使用給定的 weights 和 biases 初始化神經網路。
        """
        self.w_im = w_im
        self.b_im = b_im
        self.w_mo = w_mo
        self.b_mo = b_mo

    def forward(self, inputs):
        """
        執行神經網路的前向傳播。

        參數:
            inputs (np.ndarray): 輸入資料。預期 shape 為 (樣本數量, 輸入特徵數量)。

        回傳:
            np.ndarray: 神經網路的輸出。Shape 為 (樣本數量, 輸出特徵數量)。
        """
        # 計算傳入 hidden layer 的輸入
        # np.dot 執行 matrix multiplication (矩陣乘法)
        hidden_layer_input = np.dot(inputs, self.w_im) + self.b_im

        # 對 hidden layer 的輸入應用 sigmoid activation function
        hidden_layer_output = sigmoid(hidden_layer_input)

        # 計算傳入 output layer 的輸入
        output_layer_input = np.dot(hidden_layer_output, self.w_mo) + self.b_mo

        # 在這個範例 (regression) 中，我們不在輸出應用 activation。
        # 如果是 classification，這裡可能會使用 sigmoid 或 softmax。
        final_output = output_layer_input

        return final_output

# --- 4. 準備輸入資料 (來自截圖) ---
# 為 x_0 和 x_1 創建 100 個樣本
# np.arange 創建從 start 到 stop (不包含 stop) 的值，步長為 step。
# 截圖中的範圍 [-1.0, 1.0, 0.2] 不會產生 100 個點。
# 我們使用 linearly spaced (線性等間隔) 的方式產生 100 個點。
x_0 = np.linspace(-1.0, 1.0, 100)
x_1 = np.linspace(-1.0, 1.0, 100)

# 將 x_0 和 x_1 合併為一個輸入陣列
# Shape 應為 (樣本數量, 輸入特徵數量)
inputs = np.vstack((x_0, x_1)).T # Transpose (轉置) 為 shape (100, 2)

# 初始化一個用於儲存預期輸出的陣列 (Z)，雖然在前向傳播中未使用
# Z = np.zeros(100) # 在前向傳播計算中並未直接使用

# --- 5. 創建 NN 實例並執行前向傳播 ---
# 實例化神經網路
nn = ThreeLayerNN(w_im, b_im, w_mo, b_mo)

# 使用輸入資料執行前向傳播
predictions = nn.forward(inputs)

# --- 6. 輸出結果 (可選) ---
print("Shape of inputs (輸入資料的 shape):", inputs.shape)
print("Shape of predictions (預測結果的 shape):", predictions.shape)
print("\nFirst 5 predictions (前 5 個預測結果):\n", predictions[:5])
print("\nLast 5 predictions (後 5 個預測結果):\n", predictions[-5:])


