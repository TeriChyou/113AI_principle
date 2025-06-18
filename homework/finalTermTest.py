import numpy as np
import pandas as pd # 確保已引入 pandas
import sys
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 仍需 train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # 確保 LabelEncoder 在這裡被引入
# Data Preperation

## 1.1 Load the file
df = pd.read_csv('irisdata/iris.data', header=None)

## 1.2 Tag the data
X_data = df.iloc[:, 0:4].values # 輸入特徵
y_labels_str = df.iloc[:, 4].values # 原始字串標籤 (用於 get_dummies 和 stratify)

## 1.3-1 Use pandas.get_dummies 進行 One-Hot Encoding
y_onehot = pd.get_dummies(y_labels_str).values # 直接將字串標籤轉換為 One-Hot 編碼的 NumPy 陣列
## 1.3-2 標籤數值化 0, 1, 2
# 這裡保留 LabelEncoder 的實例化和 fit，目的是為了後續的 inverse_transform
label_encoder = LabelEncoder()
label_encoder.fit(y_labels_str) # 讓 label_encoder 學習標籤的對應關係

# 1.4 訓練集與測試集分割
# stratify 參數應使用原始的字串標籤 (y_labels_str)，以確保各類別在分割後的集合中比例一致
X_train, X_test, y_train_onehot, y_test_onehot = train_test_split(
    X_data, y_onehot, test_size=0.2, random_state=42, stratify=y_labels_str # stratify 使用原始字串標籤
)

# 1.5 特徵標準化 (順序不變，只是 X_train/X_test 現在來自新的 split)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 為了方便後續評估 (例如準確度計算)，我們可以從 one-hot 編碼的 y_train_onehot/y_test_onehot 中獲取整數標籤
y_train_int = np.argmax(y_train_onehot, axis=1)
y_test_int = np.argmax(y_test_onehot, axis=1) # 這裡的 np.argmax 會根據 get_dummies 的順序決定 0, 1, 2 對應哪個類別


print("資料準備完成:")
print(f"訓練集特徵形狀 (X_train_scaled): {X_train_scaled.shape}")
print(f"訓練集標籤形狀 (y_train_onehot): {y_train_onehot.shape}")
print(f"測試集特徵形狀 (X_test_scaled): {X_test_scaled.shape}")
print(f"測試集標籤形狀 (y_test_onehot): {y_test_onehot.shape}")
print("-" * 30)


# Model Define and Consts

# 2.1 Network Structure Consts
INPUT_SIZE = X_train_scaled.shape[1] # 4 Inputs
HIDDEN1_SIZE = 25 # First Layer 
HIDDEN2_SIZE = 25 # Second Layer
OUTPUT_SIZE = y_train_onehot.shape[1] # 3 Outputs

# 2.2 Activation Funcs la
def relu(x):
  return np.maximum(0, x)

def softmax(x):
  exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # 減去最大值避免數值溢出
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 2.3 Loss Function: Cross-Entropy 
def cross_entropy_loss(y_pred, y_true):
  epsilon = 1e-10 # 小的常數避免 log(0)
  # y_pred = np.clip(y_pred, epsilon, 1. - epsilon) # 限制預測值在 (epsilon, 1-epsilon) 之間
  loss = -np.sum(y_true * np.log(y_pred + epsilon)) / y_pred.shape[0] # 除以樣本數取平均
  return loss

# 2.4 Init weights and bias
# 使用 He Initialization (適用於 ReLU) 簡化版本，乘上 np.sqrt(2 / input_units)
# 這裡使用 np.random.randn 乘上一個小值作為初始化
np.random.seed(42) # 設置隨機種子確保可重現性

W1 = np.random.randn(INPUT_SIZE, HIDDEN1_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
b1 = np.zeros(HIDDEN1_SIZE)

W2 = np.random.randn(HIDDEN1_SIZE, HIDDEN2_SIZE) * np.sqrt(2.0 / HIDDEN1_SIZE)
b2 = np.zeros(HIDDEN2_SIZE)

W3 = np.random.randn(HIDDEN2_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / HIDDEN2_SIZE)
b3 = np.zeros(OUTPUT_SIZE)

print("網路參數初始化完成:")
print(f"W1 shape: {W1.shape}, b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}, b2 shape: {b2.shape}")
print(f"W3 shape: {W3.shape}, b3 shape: {b3.shape}")
print("-" * 30)

# --- 3. 前向傳播函數 (會儲存中間結果供反向傳播使用) ---
# 這個函數將在後面詳細實作，先定義框架
def forward_propagation(X_batch, params):
    W1, b1, W2, b2, W3, b3 = params

    # 第一層 (隱藏層1)
    H1_linear = np.dot(X_batch, W1) + b1
    H1_activated = relu(H1_linear)

    # 第二層 (隱藏層2)
    H2_linear = np.dot(H1_activated, W2) + b2
    H2_activated = relu(H2_linear)

    # 輸出層
    Output_linear = np.dot(H2_activated, W3) + b3
    Output_probabilities = softmax(Output_linear)

    # 返回激活值和中間結果 (linear output) 供反向傳播使用
    # 將所有中間結果打包成一個字典，方便管理
    cache = {
        'X': X_batch,
        'H1_linear': H1_linear, 'H1_activated': H1_activated,
        'H2_linear': H2_linear, 'H2_activated': H2_activated,
        'Output_linear': Output_linear, 'Output_probabilities': Output_probabilities
    }
    return Output_probabilities, cache

# --- 4. 反向傳播函數 (將推導並在後續實作) ---
# 這個函數將在後面詳細實作，先定義框架
def backward_propagation(y_true_batch, y_pred_batch, params, cache):
    # 獲取參數
    W1, b1, W2, b2, W3, b3 = params
    
    # 獲取前向傳播的中間結果
    X_batch = cache['X']
    H1_activated = cache['H1_activated']
    H1_linear = cache['H1_linear'] # 需要 linear output 來計算 ReLU 梯度
    H2_activated = cache['H2_activated']
    H2_linear = cache['H2_linear'] # 需要 linear output 來計算 ReLU 梯度
    Output_probabilities = cache['Output_probabilities']

    m = X_batch.shape[0] # 當前批次的樣本數

    # 輸出層梯度 (Softmax + Cross-Entropy 的簡化梯度)
    dOutput_linear = (Output_probabilities - y_true_batch) / m

    # 梯度 W3, b3
    dW3 = np.dot(H2_activated.T, dOutput_linear)
    db3 = np.sum(dOutput_linear, axis=0)

    # 隱藏層2梯度 (ReLU 激活)
    dH2_activated = np.dot(dOutput_linear, W3.T)
    # ReLU 導數: 如果 H2_linear > 0 則為 1，否則為 0
    dH2_linear = dH2_activated * (H2_linear > 0) # 應用 ReLU 導數

    # 梯度 W2, b2
    dW2 = np.dot(H1_activated.T, dH2_linear)
    db2 = np.sum(dH2_linear, axis=0)

    # 隱藏層1梯度 (ReLU 激活)
    dH1_activated = np.dot(dH2_linear, W2.T)
    # ReLU 導數: 如果 H1_linear > 0 則為 1，否則為 0
    dH1_linear = dH1_activated * (H1_linear > 0) # 應用 ReLU 導數

    # 梯度 W1, b1
    dW1 = np.dot(X_batch.T, dH1_linear)
    db1 = np.sum(dH1_linear, axis=0)

    gradients = {
        'dW1': dW1, 'db1': db1,
        'dW2': dW2, 'db2': db2,
        'dW3': dW3, 'db3': db3
    }
    return gradients

# --- 5. 優化器實現 (SGD 和 AdaGrad) ---
# 這個函數將在後面詳細實作，先定義框架
def update_parameters(params, gradients, learning_rate, optimizer_type='SGD', s_params=None, epsilon=1e-8):
    W1, b1, W2, b2, W3, b3 = params
    dW1, db1, dW2, db2, dW3, db3 = (gradients[k] for k in ['dW1', 'db1', 'dW2', 'db2', 'dW3', 'db3'])

    if optimizer_type == 'SGD':
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        return (W1, b1, W2, b2, W3, b3), None # SGD不需要返回s_params

    elif optimizer_type == 'AdaGrad': # 此AGD演算法的來源為:
        sW1, sb1, sW2, sb2, sW3, sb3 = (s_params[k] for k in ['sW1', 'sb1', 'sW2', 'sb2', 'sW3', 'sb3'])

        sW1 += dW1**2
        sb1 += db1**2
        sW2 += dW2**2
        sb2 += db2**2
        sW3 += dW3**2
        sb3 += db3**2

        W1 -= learning_rate * dW1 / (np.sqrt(sW1) + epsilon)
        b1 -= learning_rate * db1 / (np.sqrt(sb1) + epsilon)
        W2 -= learning_rate * dW2 / (np.sqrt(sW2) + epsilon)
        b2 -= learning_rate * db2 / (np.sqrt(sb2) + epsilon)
        W3 -= learning_rate * dW3 / (np.sqrt(sW3) + epsilon)
        b3 -= learning_rate * db3 / (np.sqrt(sb3) + epsilon)

        s_params_updated = {
            'sW1': sW1, 'sb1': sb1,
            'sW2': sW2, 'sb2': sb2,
            'sW3': sW3, 'sb3': sb3
        }
        return (W1, b1, W2, b2, W3, b3), s_params_updated
    else:
        raise ValueError("Unsupported optimizer type. Choose 'SGD' or 'AdaGrad'.")

# --- 輔助函數: 評估準確度 ---
def predict(X, params):
    probabilities, _ = forward_propagation(X, params)
    return np.argmax(probabilities, axis=1) # 返回預測類別的整數標籤

def calculate_accuracy(y_pred_int, y_true_int):
    return np.mean(y_pred_int == y_true_int)


# --- 6. 訓練迴圈與評估 ---

# 訓練參數
EPOCHS = 1000 # 訓練輪數
BATCH_SIZE = 32 # 批次大小
LEARNING_RATE = 0.01 # 學習率
OPTIMIZER_TYPE = 'SGD' # 可選 'SGD' 或 'AdaGrad'

# 初始化參數 (傳遞給 update_parameters)
current_params = [W1, b1, W2, b2, W3, b3]

# AdaGrad 相關參數 (如果使用 AdaGrad)
if OPTIMIZER_TYPE == 'AdaGrad':
    s_params = {
        'sW1': np.zeros_like(W1), 'sb1': np.zeros_like(b1),
        'sW2': np.zeros_like(W2), 'sb2': np.zeros_like(b2),
        'sW3': np.zeros_like(W3), 'sb3': np.zeros_like(b3)
    }
else:
    s_params = None # SGD不需要

# 記錄訓練進度
train_loss_history = []
test_loss_history = []
train_accuracy_history = []
test_accuracy_history = []
epoch_indices = []

# --- 設置 Matplotlib 互動模式和初始化圖表 ---
plt.ion() # 開啟互動模式
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # 創建圖形和兩個子圖的軸

# 初始化圖表配置 (在迴圈外一次性設置標題、標籤和網格)
ax1.set_title('Loss Curve')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss (Cross-Entropy)')
ax1.grid(True)

ax2.set_title('Accuracy Curve')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid(True)

print(f"\n開始訓練 (Optimizer: {OPTIMIZER_TYPE}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE})")
print("-" * 60)

for epoch in range(EPOCHS):
    # 打亂訓練資料索引，用於批次訓練
    shuffled_indices = np.arange(X_train_scaled.shape[0])
    np.random.shuffle(shuffled_indices)

    # 批次訓練
    for i in range(0, X_train_scaled.shape[0], BATCH_SIZE):
        batch_indices = shuffled_indices[i : i + BATCH_SIZE]
        X_batch = X_train_scaled[batch_indices]
        y_true_batch = y_train_onehot[batch_indices]

        # 前向傳播
        y_pred_batch, cache = forward_propagation(X_batch, current_params)

        # 反向傳播 (計算梯度)
        gradients = backward_propagation(y_true_batch, y_pred_batch, current_params, cache)

        # 更新參數
        if OPTIMIZER_TYPE == 'AdaGrad':
            current_params, s_params = update_parameters(current_params, gradients, LEARNING_RATE, OPTIMIZER_TYPE, s_params)
        else:
            current_params, _ = update_parameters(current_params, gradients, LEARNING_RATE, OPTIMIZER_TYPE)

    # 每個 Epoch 結束時評估並記錄
    # 訓練集評估
    train_y_pred_probs, _ = forward_propagation(X_train_scaled, current_params)
    train_loss = cross_entropy_loss(train_y_pred_probs, y_train_onehot)
    train_accuracy = calculate_accuracy(np.argmax(train_y_pred_probs, axis=1), y_train_int)

    # 測試集評估
    test_y_pred_probs, _ = forward_propagation(X_test_scaled, current_params)
    test_loss = cross_entropy_loss(test_y_pred_probs, y_test_onehot)
    test_accuracy = calculate_accuracy(np.argmax(test_y_pred_probs, axis=1), y_test_int)

    # 記錄歷史
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    train_accuracy_history.append(train_accuracy)
    test_accuracy_history.append(test_accuracy)
    epoch_indices.append(epoch)

    # --- 動態更新圖形 ---
    # 清除舊的繪圖內容
    ax1.clear()
    ax2.clear()

    # 重新繪製損失曲線
    ax1.plot(epoch_indices, train_loss_history, label='Train Loss', color='blue')
    ax1.plot(epoch_indices, test_loss_history, label='Test Loss', color='red')
    # 重新設定標題、標籤和圖例 (因為 clear() 會移除這些)
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Cross-Entropy)')
    ax1.legend()
    ax1.grid(True)

    # 重新繪製準確度曲線
    ax2.plot(epoch_indices, train_accuracy_history, label='Train Accuracy', color='blue')
    ax2.plot(epoch_indices, test_accuracy_history, label='Test Accuracy', color='red')
    # 重新設定標題、標籤和圖例
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # 將更新繪製到畫布上
    fig.canvas.draw()
    # 強制事件處理，讓圖形立即刷新
    fig.canvas.flush_events()
    # 短暫暫停，讓圖形有時間顯示 (時間可以調整)
    plt.pause(0.001) 

    # 打印進度條和實時資訊 (每 100 epoch 打印一次詳細資訊)
    if (epoch + 1) % 100 == 0 or epoch == 0 or (epoch + 1) == EPOCHS:
        progress = int((epoch + 1) / EPOCHS * 30)
        bar = '[' + '#' * progress + '-' * (30 - progress) + ']'
        sys.stdout.write(f'\rEpoch: {epoch + 1}/{EPOCHS} {bar} '
                         f'Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | '
                         f'Test Loss: {test_loss:.4f} Acc: {test_accuracy:.4f}')
        sys.stdout.flush()

print("\n" + "-" * 60)
print("訓練完成！")

# --- 最終評估 ---
final_train_accuracy = calculate_accuracy(predict(X_train_scaled, current_params), y_train_int)
final_test_accuracy = calculate_accuracy(predict(X_test_scaled, current_params), y_test_int)

print(f"\n最終訓練準確度: {final_train_accuracy * 100:.2f}%")
print(f"最終測試準確度: {final_test_accuracy * 100:.2f}%")

# --- 關閉互動模式並顯示最終圖形 ---
# 訓練結束後，關閉互動模式，並顯示最終的圖形 (讓圖形窗口保持打開)
plt.ioff()
plt.show()

print("\n預測部分範例:")
sample_indices = np.random.choice(len(X_test_scaled), 5, replace=False) # 隨機取 5 個測試樣本
for idx in sample_indices:
    sample_X = X_test_scaled[idx].reshape(1, -1) # 轉成 (1, 4) 的 shape
    true_label_int = y_test_int[idx]
    true_label_str = label_encoder.inverse_transform(np.array([true_label_int]))[0]
    
    predicted_probs, _ = forward_propagation(sample_X, current_params)
    predicted_label_int = np.argmax(predicted_probs, axis=1)[0]
    predicted_label_str = label_encoder.inverse_transform(np.array([predicted_label_int]))[0]

    print(f"樣本 {idx+1}: 真實類別: {true_label_str} ({true_label_int}), "
          f"預測機率: {predicted_probs[0]}, 預測類別: {predicted_label_str} ({predicted_label_int})")