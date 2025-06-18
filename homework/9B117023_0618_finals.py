# Iris AdaGrad Neural Network Training Programming Modification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler # 用於特徵標準化 => Easier for Standardization
from sklearn.preprocessing import LabelEncoder # 用於將數值標籤轉回字串，用於最終預測顯示 => For final prediction display
import sys

std_schno = '9B117023' # stdno

# --- 1. 載入與預處理資料 ---
# LOAD DATA
iris_data_df = pd.read_csv('iris.data', header=None)

# Eigenvalue & Tag Split
input_data_raw = iris_data_df.iloc[:, 0:4].values # 輸入特徵 (原始值)
y_labels_str = iris_data_df.iloc[:, 4].values # 原始字串標籤

# 將正確答案做 one-hot 編碼 (使用 pandas.get_dummies)
correct_data_onehot = pd.get_dummies(y_labels_str).values.astype(int) # to make display 0 and 1

# 因為之後要將數值標籤轉回字串，保留 LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y_labels_str)


# 將原始資料的前四個欄位做標準化 X_std=(X-X_mean)/X_stddev
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data_raw) # 對所有原始資料進行標準化

# 訓練資料與測試資料分割 (各品種留20筆測試，其餘訓練)
# 使用 sklearn.model_selection.train_test_split 進行分層抽樣
# 因為 correct_data_onehot 已經是 One-Hot 編碼，stratify 需要原始的整數標籤
y_labels_int = np.argmax(correct_data_onehot, axis=1) # 從 One-Hot 轉回整數標籤給 stratify

# 分割資料的函式 => 
# 各品種前 20 筆當測試資料，後 30 筆當訓練資料
# X_test (測試特徵: 20筆 Setosa + 20筆 Versicolor + 20筆 Virginica = 共 60 筆)
X_test = np.vstack((
    input_data_scaled[0:20],      # Setosa 的前 20 筆
    input_data_scaled[50:70],     # Versicolor 的前 20 筆
    input_data_scaled[100:120]    # Virginica 的前 20 筆
))

# X_train (訓練特徵: 30筆 Setosa + 30筆 Versicolor + 30筆 Virginica = 共 90 筆)
X_train = np.vstack((
    input_data_scaled[20:50],     # Setosa 的後 30 筆
    input_data_scaled[70:100],    # Versicolor 的後 30 筆
    input_data_scaled[120:150]    # Virginica 的後 30 筆
))

# y_test_onehot (測試標籤: 20筆 Setosa + 20筆 Versicolor + 20筆 Virginica = 共 60 筆)
y_test_onehot = np.vstack((
    correct_data_onehot[0:20],
    correct_data_onehot[50:70],
    correct_data_onehot[100:120]
))

# y_train_onehot (訓練標籤: 30筆 Setosa + 30筆 Versicolor + 30筆 Virginica = 共 90 筆)
y_train_onehot = np.vstack((
    correct_data_onehot[20:50],
    correct_data_onehot[70:100],
    correct_data_onehot[120:150]
))
# 確保 n_train 和 n_test 與分割後的數據形狀一致
n_train = X_train.shape[0] # 訓練資料的樣本數 (90)
n_test = X_test.shape[0]   # 測試資料的樣本數 (60)

print(f"資料準備完成:")
print(f"原始資料總數: {len(input_data_raw)}")
print(f"訓練集特徵形狀 (X_train): {X_train.shape}")
print(f"訓練集標籤形狀 (y_train_onehot): {y_train_onehot.shape}")
print(f"測試集特徵形狀 (X_test): {X_test.shape}")
print(f"測試集標籤形狀 (y_test_onehot): {y_test_onehot.shape}")
print("-" * 30)

# Print => 請將最後一個欄位鳶尾花品名做one hot 編碼: 100,010,001三種,列印 [V]
print(f"One-Hot 編碼後的資料:")
print(correct_data_onehot)
print("-" * 30)

# Print => X_std=(X-X_mean)/X_stddev [V]
print(f"標準化後的特徵 (input_data_scaled):")
print(input_data_scaled)
print("-" * 30)

# --- 各個設定值---
n_in = 4     # 輸入層的神經元數量
# 神經元數量based on 9B11"7023"
n_mid_1 = 70 # 第一個中間層的神經元數量 
n_mid_2 = 23 # 第二個中間層的神經元數量
n_out = 3    # 輸出層的神經元數量

wb_width = 0.1  # 權重與偏向量的範圍 
eta = 0.01      # learning rate
EPOCHS = 600    # 訓練週期 (Req.600)
BATCH_SIZE = 10 # 批次大小 (Req.10)
interval = 10  # Progress Interval => 這樣視覺上看起來跑得比較順(阿不然卡卡的)

# 優化器固定為 AdaGrad
OPTIMIZER_TYPE = 'AdaGrad'


# --- 各層的 Class 定義  ---

class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n) 

        # AdaGrad 相關參數的初始化
        self.h_w = np.zeros_like(self.w) + 1e-8
        self.h_b = np.zeros_like(self.b) + 1e-8
        
    def update(self, eta_val): # AdaGrad 版本，不再需要 optimizer_type_val 參數
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta_val / np.sqrt(self.h_w) * self.grad_w
        
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta_val / np.sqrt(self.h_b) * self.grad_b
        
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u) # ReLU
        return self.y
    
    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1) # ReLU的微分

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 
        return self.grad_x

class OutputLayer(BaseLayer):     
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        exp_u = np.exp(u - np.max(u, axis=1, keepdims=True)) # 減去最大值提高數值穩定性
        self.y = exp_u / np.sum(exp_u, axis=1, keepdims=True) # Softmax
        return self.y

    def backward(self, t): # t 是 One-Hot 編碼的真實標籤
        delta = self.y - t # Softmax 結合 Cross-Entropy Loss 的簡化梯度
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 
        return self.grad_x

# --- According to the node number to make the layers ---
middle_layer_1 = MiddleLayer(n_in, n_mid_1) # 輸入層(4) -> 第一個中間層(70)
middle_layer_2 = MiddleLayer(n_mid_1, n_mid_2) # 第一個中間層(70) -> 第二個中間層(23)
output_layer = OutputLayer(n_mid_2, n_out) # 第二個中間層(23) -> 輸出層(3)


# --- forward propa/ back propa/ update weight and bias funcitons  ---

def forward_propagation(x_input):
    """執行前向傳播，計算各層輸出"""
    middle_layer_1.forward(x_input)
    middle_layer_2.forward(middle_layer_1.y)
    output_layer.forward(middle_layer_2.y)

def backpropagation(t_true):
    """執行反向傳播，計算各層權重與偏差的梯度"""
    output_layer.backward(t_true)
    middle_layer_2.backward(output_layer.grad_x)
    middle_layer_1.backward(middle_layer_2.grad_x)

def update_wb_all_layers(eta_val): # AdaGrad 版本，update 方法不需要 optimizer_type_val 參數
    """更新所有層的權重與偏差"""
    middle_layer_1.update(eta_val)
    middle_layer_2.update(eta_val)
    output_layer.update(eta_val)

# --- 計算交叉熵誤差  ---
def get_error(t_true, num_samples_in_batch):
    return -np.sum(t_true * np.log(output_layer.y + 1e-7)) / num_samples_in_batch


# --- 訓練迴圈與評估 ---

# 記錄訓練進度
train_loss_history = []
test_loss_history = []
train_accuracy_history = []
test_accuracy_history = []
epoch_indices = []

# --- 設置 Matplotlib 互動模式和初始化圖表 ---
plt.ion() # 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # 

# 初始化圖表配置
ax1.set_title('Loss Curve')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss (Cross-Entropy)')
ax1.grid(True)

ax2.set_title('Accuracy Curve')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid(True)


print(f" - 開始訓練 (優化器: {OPTIMIZER_TYPE}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, 學習率: {eta})")
print("-" * 60)

# -- 開始訓練 --
n_batch = n_train // BATCH_SIZE # 每 1 epoch 的批次數量

for epoch in range(EPOCHS):
    # 打亂訓練資料索引，用於批次訓練
    index_random = np.arange(n_train)
    np.random.shuffle(index_random)

    # 批次訓練
    for j in range(n_batch):
        # 取出小批次
        mb_index = index_random[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
        x_batch = X_train[mb_index, :] 
        t_batch = y_train_onehot[mb_index, :] 
        # 前向傳播與反向傳播
        forward_propagation(x_batch)
        backpropagation(t_batch)
        
        # 更新權重與偏值 (AdaGrad 版本)
        update_wb_all_layers(eta)

    # -- 每個 Epoch 結束時計算並記錄誤差和準確度 --
    # 計算訓練集誤差和準確度
    forward_propagation(X_train) 
    error_train = get_error(y_train_onehot, n_train) 
    train_predicted_labels_int = np.argmax(output_layer.y, axis=1)
    train_actual_labels_int = np.argmax(y_train_onehot, axis=1) 
    accuracy_train = np.mean(train_predicted_labels_int == train_actual_labels_int)

    # 計算測試集誤差和準確度
    forward_propagation(X_test)
    error_test = get_error(y_test_onehot, n_test) 
    test_predicted_labels_int = np.argmax(output_layer.y, axis=1)
    test_actual_labels_int = np.argmax(y_test_onehot, axis=1) 
    accuracy_test = np.mean(test_predicted_labels_int == test_actual_labels_int)
    
    
    # 記錄歷史
    train_loss_history.append(error_train)
    test_loss_history.append(error_test)
    train_accuracy_history.append(accuracy_train)
    test_accuracy_history.append(accuracy_test)
    epoch_indices.append(epoch)

    # --- 動態更新圖形 ---
    ax1.clear(); ax2.clear()
    ax1.plot(epoch_indices, train_loss_history, label='Train Loss', color='blue')
    ax1.plot(epoch_indices, test_loss_history, label='Test Loss', color='red')
    ax1.set_title('Loss Curve'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss (Cross-Entropy)'); ax1.legend(); ax1.grid(True)

    ax2.plot(epoch_indices, train_accuracy_history, label='Train Accuracy', color='blue')
    ax2.plot(epoch_indices, test_accuracy_history, label='Test Accuracy', color='red')
    ax2.set_title('Accuracy Curve'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True)

    fig.canvas.draw(); fig.canvas.flush_events()
    plt.pause(0.001) 

    # 打印進度條和實時資訊 (每隔 interval 打印一次，或在首尾 epoch 打印)
    if (epoch + 1) % interval == 0 or epoch == 0 or (epoch + 1) == EPOCHS:
        progress_val = int((epoch + 1) / EPOCHS * 30)
        bar_val = '[' + '#' * progress_val + '-' * (30 - progress_val) + ']'
        sys.stdout.write(f'\rEpoch: {epoch + 1}/{EPOCHS} {bar_val} '
                         f'Train Loss: {error_train:.4f} Acc: {accuracy_train:.4f} | '
                         f'Test Loss: {error_test:.4f} Acc: {accuracy_test:.4f}')
        sys.stdout.flush()

print("\n" + "-" * 60)
print("Done Training！")

# --- 最終評估 ---
print(f"Final Training Accuracy: {accuracy_train * 100:.2f}%")
print(f"Final Testing Accuracy: {accuracy_test * 100:.2f}%")

# --- 關閉互動模式並顯示最終圖形 ---
plt.ioff()
plt.show()

# Print => 將原始測試資料品名 one hot encoding 矩陣列印出來 (20x3筆)
print(" - 原始測試資料品名 One-Hot 編碼矩陣 (y_test_onehot):")
print(y_test_onehot)
print("-" * 30)

# Print => 將最後放入測試資料(input_test)到output_layer的預測結果列印出來(20x3筆)
# use forward propagation to make sure output_layer.y is latest. :O
forward_propagation(X_test)
print("測試資料 (input_test) 的最終預測結果 (output_layer.y):")
print(output_layer.y)
print("-" * 30)

# Bonus 隨機抓一下資料做預測範例 
print("測試資料預測範例 (隨機取樣5筆):")
sample_indices = np.random.choice(n_test, 5, replace=False) # 隨機取 5 個測試樣本
for idx in sample_indices:
    sample_X = X_test[idx].reshape(1, -1)
    
    # 獲取真實標籤
    true_label_onehot = y_test_onehot[idx]
    true_label_int = np.argmax(true_label_onehot)
    true_label_str = label_encoder.inverse_transform(np.array([true_label_int]))[0]
    
    # 執行前向傳播獲取預測結果
    forward_propagation(sample_X)
    predicted_probs = output_layer.y[0] # 獲取輸出層的預測機率 (因為是單樣本，取第一行)
    formatted_probs_list = [f"{p*100:.2f}%" for p in predicted_probs]
    predicted_probs_str_formatted = f"[{', '.join(formatted_probs_list)}]"
    predicted_label_int = np.argmax(predicted_probs)
    predicted_label_str = label_encoder.inverse_transform(np.array([predicted_label_int]))[0]

    print(f"樣本索引 {idx}: 真實類別: {true_label_str} ({true_label_int}), "
          f"預測機率: {predicted_probs_str_formatted}, 預測類別: {predicted_label_str} ({predicted_label_int})")
print("-" * 30)