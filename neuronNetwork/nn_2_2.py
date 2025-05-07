from modules import *
# import matplotlib.pyplot as plt # 可選，用於繪圖或可視化

# --- 1. 定義 Sigmoid Activation Function ---
def sigmoid(x):
  """
  Sigmoid 激活函數。
  它將輸入值壓縮到 0 到 1 之間。
  """
  return 1 / (1 + np.exp(-x))

# --- 2. 初始化 Weights 和 Biases (根據新的圖) ---
# 從 input layer 到 hidden layer 的 weights (2 個輸入, 3 個隱藏節點)
# Shape 是 (輸入數量, 隱藏節點數量) = (2, 3)
w_im = np.array([[0.1, 0.2, 0.3],
                 [0.4, 0.55, 0.6]])

# Biases for the hidden layer (3 個隱藏節點)
# Shape 是 (隱藏節點數量,) = (3,)
b_im = np.array([0.6, 0.3, 0.4])

# 從 hidden layer 到 output layer 的 weights (3 個隱藏節點, 1 個輸出節點)
# Shape 是 (隱藏節點數量, 輸出節點數量) = (3, 1)
w_mo = np.array([[0.4],
                 [0.5],
                 [-0.3]])

# Bias for the output layer (1 個輸出節點)
# Shape 是 (輸出節點數量,) = (1,)
b_mo = np.array([0.2])

# --- 3. 定義 Neural Network Class ---
class ThreeLayerNN_MoreHidden:
    def __init__(self, w_im, b_im, w_mo, b_mo):
        """
        使用給定的 weights 和 biases 初始化神經網路 (3 個隱藏節點)。
        """
        self.w_im = w_im
        self.b_im = b_im
        self.w_mo = w_mo
        self.b_mo = b_mo

    def forward(self, inputs):
        """
        執行神經網路的前向傳播 (3 個隱藏節點)。

        Args:
            inputs (np.ndarray): 輸入資料。預期 shape (樣本數量, 輸入特徵數量)。
                                 對於單個樣本，shape 應為 (1, 2)。

        Returns:
            np.ndarray: 神經網路的輸出。Shape (樣本數量, 輸出特徵數量)。
                       在這個結構中，輸出特徵數量是 1。
        """
        # 計算傳入 hidden layer 的輸入
        # inputs shape: (N, 2), w_im shape: (2, 3) -> np.dot 得到 (N, 3)
        # b_im shape: (3,) -> broadcasting 加到 (N, 3) 上
        hidden_layer_input = np.dot(inputs, self.w_im) + self.b_im

        # 對 hidden layer 的輸入應用 sigmoid activation function
        # hidden_layer_input shape: (N, 3) -> sigmoid 得到 (N, 3)
        hidden_layer_output = sigmoid(hidden_layer_input)

        # 計算傳入 output layer 的輸入
        # hidden_layer_output shape: (N, 3), w_mo shape: (3, 1) -> np.dot 得到 (N, 1)
        # b_mo shape: (1,) -> broadcasting 加到 (N, 1) 上
        output_layer_input = np.dot(hidden_layer_output, self.w_mo) + self.b_mo

        # output layer 沒有 activation function
        final_output = output_layer_input

        return final_output

# --- 4. 準備輸入資料 (使用 linspace 生成 100x100 的組合) ---
# 我們可以生成一系列輸入來測試網路
x_0_test = np.linspace(-1.0, 1.0, 100)
x_1_test = np.linspace(-1.0, 1.0, 100)

# 為了使用 class 的 forward 方法處理批次輸入，我們將所有組合創建為一個大的輸入陣列
# 方法一：使用迴圈和 vstack/reshape (如果記憶體允許)
# test_inputs = []
# for i in range(100):
#     for j in range(100):
#         test_inputs.append([x_0_test[i], x_1_test[j]])
# test_inputs = np.array(test_inputs) # Shape (10000, 2)

# 方法二：使用 NumPy 的 meshgrid 和 reshape (更有效率)
X0, X1 = np.meshgrid(x_0_test, x_1_test)
# 將網格展平並合併為輸入陣列
test_inputs = np.vstack([X0.ravel(), X1.ravel()]).T # Shape (10000, 2)


# --- 5. 創建 NN 實例並執行前向傳播 ---
# 實例化新的神經網路
nn_more_hidden = ThreeLayerNN_MoreHidden(w_im, b_im, w_mo, b_mo)

# 使用測試輸入資料執行前向傳播
predictions = nn_more_hidden.forward(test_inputs)

# --- 6. 打印結果 (可選) ---
print("Shape of test_inputs:", test_inputs.shape)
print("Shape of predictions:", predictions.shape)
print("\nFirst 10 predictions:\n", predictions[:10])
print("\nLast 10 predictions:\n", predictions[-10:])

# --- 如何使用單一樣本進行測試 ---
# single_input = np.array([[0.5, -0.2]]) # 注意雙層中括號
# single_prediction = nn_more_hidden.forward(single_input)
# print("\nPrediction for single input [0.5, -0.2]:", single_prediction)