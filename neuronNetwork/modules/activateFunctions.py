import numpy as np

class AF:
    # --- ReLU ---
    def relu_layer(x, w, b):
        """
        計算一層的輸出，使用 ReLU (Rectified Linear Unit) 激活函數。
        ReLU: f(u) = max(0, u)
        """
        u = np.dot(x, w) + b
        return np.maximum(0, u)

    # --- Leaky ReLU ---
    # alpha 是一個小的正數 (例如 0.01)，用於負數部分
    def leaky_relu_layer(x, w, b, alpha=0.01):
        """
        計算一層的輸出，使用 Leaky ReLU 激活函數。
        Leaky ReLU: f(u) = u if u > 0, else alpha * u
        """
        u = np.dot(x, w) + b
        return np.where(u > 0, u, alpha * u) # 使用 np.where 實現條件判斷

    # --- PReLU (Parametric ReLU) ---
    # PReLU 的 alpha 是可學習的參數，這裡以函數參數表示
    # 在實際訓練中，alpha 會作為模型參數進行優化
    def prelu_layer(x, w, b, alpha=0.25): # 這裡設定一個範例 alpha 值
        """
        計算一層的輸出，使用 PReLU (Parametric ReLU) 激活函數。
        PReLU: f(u) = u if u > 0, else alpha * u (alpha 是可學習參數)
        """
        u = np.dot(x, w) + b
        return np.where(u > 0, u, alpha * u)

    # --- RReLU (Randomized ReLU) ---
    # RReLU 的 alpha 在訓練時是從給定範圍內隨機抽樣的，在測試時固定為範圍的平均值。
    # 這個特性在一個靜態函數中難以完整呈現。這裡提供一個簡化版本，類似 Leaky ReLU，
    # 如果需要訓練階段的隨機性，則需要在訓練迴圈中處理 alpha 的抽樣。
    # 在測試時通常固定 alpha，所以這個函數在測試時行為類似 Leaky ReLU。
    # 這裡使用一個固定的 alpha 值來表示測試階段或簡化概念。
    def rrelu_layer(x, w, b, alpha=0.1): # 這裡使用一個範例固定 alpha 值
        """
        計算一層的輸出，使用 RReLU (Randomized ReLU) 激活函數的簡化表示 (測試階段行為)。
        RReLU 在訓練時負數部分的斜率是隨機的。
        """
        u = np.dot(x, w) + b
        return np.where(u > 0, u, alpha * u) # 簡化表示，使用固定 alpha


    # --- Sigmoid ---
    def sigmoid_layer(x, w, b): # => Sigmoid ()
        """
        計算一層的輸出，使用 Sigmoid 激活函數。
        Sigmoid: f(u) = 1 / (1 + exp(-u))
        """
        u = np.dot(x, w) + b
        return 1/(1+np.exp(-u))

    # --- Tanh (Hyperbolic Tangent) ---
    def tanh_layer(x, w, b):
        """
        計算一層的輸出，使用 Tanh (Hyperbolic Tangent) 激活函數。
        Tanh: f(u) = tanh(u)
        """
        u = np.dot(x, w) + b
        return np.tanh(u)

    # --- Softmax (通常用於最終輸出層，多類別分類) ---
    # 這裡也提供一個 Softmax 的版本，但要注意它的用途和 np.sum 的軸
    # 通常輸入 x 是來自前一層的輸出，shape (N, 前一層節點數)
    # w 是連接前一層到輸出層的權重，shape (前一層節點數, 輸出節點數)
    # b 是輸出層的偏差，shape (輸出節點數,)
    # u = np.dot(x, w) + b 的 shape 會是 (N, 輸出節點數)
    def softmax_layer(x, w, b):
        """
        計算一層的輸出，使用 Softmax 激活函數 (常用於最終輸出層)。
        Softmax 將輸出轉換為機率分佈。
        """
        u = np.dot(x, w) + b
        # 為了數值穩定性，減去最大值
        exp_u = np.exp(u - np.max(u, axis=-1, keepdims=True))
        # 在輸出節點維度上求和，並保持維度以便廣播
        return exp_u / np.sum(exp_u, axis=-1, keepdims=True)
