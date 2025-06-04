# NCUT113_2_AI_9B117023_0604

import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt

# 1. Data import and preprocess
df = pd.read_csv('irisdata/iris.data', header=None)

# setosa and verginica
x_setosa = df.iloc[0:50, [0,1,2,3]].values
y_setosa = df.iloc[0:50, 4].values
x_virginica = df.iloc[100:150, [0,1,2,3]].values
y_virginica = df.iloc[100:150, 4].values


# combine the data
x = np.vstack((x_setosa, x_virginica))
y = np.concatenate((y_setosa, y_virginica))


# 將 'iris-setosa' 轉為 0，'iris-virginica' 轉為 1
y = np.where(y == 'Iris-setosa', 0, 1)


# 2. 資料集分割: 每種花前15筆測試，後35筆訓練
x_train = np.empty((70, 4))
x_test = np.empty((30, 4))
y_train = np.empty(70)
y_test = np.empty(30)

# 分割 Setosa (x 的前 50 筆)
x_test[0:15] = x[0:15]
y_test[0:15] = y[0:15]
x_train[0:35] = x[15:50]
y_train[0:35] = y[15:50]

# 分割 Virginica (x 的後 50 筆)
x_test[15:30] = x[50:65]
y_test[15:30] = y[50:65]
x_train[35:70] = x[65:100]
y_train[35:70] = y[65:100]


# Function Section :)

def sigmoid(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))

def update(x_data, y_true, w, b, eta):
    y_pred = sigmoid(x_data, w, b)
    a = (y_pred - y_true) * y_pred * (1 - y_pred) # 誤差修正項
    
    for i in range(w.shape[0]):
        w[i] -= eta * (1/float(len(y_true))) * np.sum(a*x_data[:,i])
    b -= eta * (1/float(len(y_true))) * np.sum(a)
    return w, b

# 計算均方誤差 (Mean Squared Error, MSE)
def get_loss(x_data, y_true, w, b):
    y_pred = sigmoid(x_data, w, b)
    mse = np.mean((y_pred - y_true)**2)
    return mse

# Main Training progress
weights = np.ones(4)/10
bias = np.ones(1)/10
eta = 0.1
epochs = 500 # 訓練次數 500

# Error recording
train_loss_history = []
test_loss_history = []
epoch_history = []

for epoch in range(epochs):
    weights, bias = update(x_train, y_train, weights, bias, eta)
    
    # Calculation of training error 
    train_loss = get_loss(x_train, y_train, weights, bias)
    test_loss = get_loss(x_test, y_test, weights, bias)
    
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    epoch_history.append(epoch)
    
    # Progress bar and error displaying.
    progress = int((epoch + 1) / epochs * 30)
    bar = '[' + '#' * progress + '-' * (30 - progress) + ']'
    sys.stdout.write(f'\rTraining Progress: {bar} {epoch + 1}/{epochs} '
                     f'Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f}')
    sys.stdout.flush()
    time.sleep(0.01)

print() # To next line after training done

# Result Printing
print('訓練後的權重 (weights)=', weights)
print('訓練後的偏差 (bias)=', bias)

test_predictions_raw = sigmoid(x_test, weights, bias)
print('\n測試資料的預測機率 (sigmoid output):')
print(test_predictions_raw)

test_predictions_labels = np.where(test_predictions_raw >= 0.5, 1, 0)
print('\n測試資料的預測類別:')
print(test_predictions_labels)
print('\n測試資料的真實類別:')
print(y_test)

# 計算並印出測試資料的準確度
accuracy = np.mean(test_predictions_labels == y_test)
print(f'\n測試資料的準確度 (Accuracy): {accuracy * 100:.2f}%')

# --- 繪製誤差記錄曲線 ---
plt.figure(figsize=(10, 6))
plt.plot(epoch_history, train_loss_history, label="Train Loss", color='blue')
plt.plot(epoch_history, test_loss_history, label="Test Loss", color='red')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training and Test Loss Curve")
plt.grid(True)
plt.show()