"""
# Author: YuQuang
# Date  : 2021/06/16
# Breif : This is a simple NN. To classification Iris.
"""
import numpy as np
import pandas as pd

"""
# 加載資料，將資料已 onehot 方式編碼
"""
def load_label(labels):
    one_hot_labels = np.zeros((labels.shape[0],2))  # 初始化陣列 ( 資料筆數 x 幾類 )
    for i in range(labels.shape[0]):                # 資料筆數
        one_hot_labels[i,labels[i]] = 1             # 那一類所在的那一行設定為 1
    return one_hot_labels

"""
# 激勵函數
"""
def sigmoid(x):
    return 1/(1+np.exp(-x)) # Sigmoid 公式

"""
# 計算值乘上權重與偏差
"""
def inner_product(x_train, w, b):
    return np.dot(x_train, w) + b

"""
# 將激勵函數與矩陣值計算
"""
def activation(x_train,w,b):
    return sigmoid(inner_product(x_train, w, b))

"""
# 前向傳播
"""
def calculate(x_train, w_list, b_list):
    val_dict={}                                         # 保存每層輸出的字典 ( 亦可用陣列替代 )
    u_1=inner_product(x_train, w_list[0], b_list[0])    # 輸入層到中間層
    y_1=sigmoid(u_1)                                    # 對輸入層到中間層的計算值做激勵函數
    
    u_2=inner_product(y_1, w_list[1], b_list[1])        # 中間層到輸出層
    y_2=sigmoid(u_2)                                    # 對中間層到輸出層的計算值做激勵函數
    
    val_dict['y_1']=y_1                                 # 返回中間層輸出方便做反向傳播
    val_dict['y_2']=y_2                                 # 返回最終輸出
    
    return val_dict
"""
# 更新權重以及偏差值 ( 反向傳播 )
"""
def update(x, y, w_list, b_list, eta):
    """
        ( 正向傳播 )
    """
    y_pred   = calculate( x, w_list, b_list )   # 前向傳播的結果
    y_1      = y_pred['y_1']                    # y_1(2, 80) 中間層輸出結果
    y_2      = y_pred['y_2']                    # y_2(2, 1) 最後輸出結果

    """
        ( 反向傳播 )
    """
    d12_d11   = 1.0
    d11_d9    = ( 1/x.shape[0] ) * ( y_2.reshape(y_2.shape[0]) - y )
    d9_d8     = y_2.reshape(y_2.shape[0]) * ( 1.0 - y_2.reshape(y_2.shape[0]) )
    d8_d7     = 1.0

    d8_d6     = y_1.T
    d8_d5     = w_list[1].T
    d5_d4     = y_1 * ( 1.0 - y_1 )
    d4_d3     = 1.0
    d4_d2     = x.T
    
    d12_d8    = d12_d11 * d11_d9 * d9_d8


    b_list[1]-= eta * np.sum( d12_d8 * d8_d7, axis=0 )
    w_list[1]-= eta * np.dot( d8_d6, d12_d8 ).reshape(w_list[1].shape)
    
    d12_d8    = d12_d11 * d11_d9 * d9_d8
    d12_d5    = np.dot( d12_d8.reshape((80, 1)), d8_d5 )
    d12_d4    = d12_d5 * d5_d4
    b_list[0]-= eta * np.sum( d12_d4*d4_d3, axis=0 )
    w_list[0]-= eta * np.dot( d4_d2, d12_d4 )

    """
        損失函數
    """
    loss = ( 1/2*x.shape[0] ) * np.sum( y_2.reshape(y_2.shape[0]) - y )**2    # 計算損失函數 此為均方差公式


    return w_list, b_list, loss # 返回更新的 權重、偏差 以及 Loss


"""
    程式進入點
"""
if __name__ == "__main__":
    df = pd.read_csv('irisdata/iris_header.csv',header=None)  # 讀取資料

    
    x = df.iloc[1:151, [0,1,2,3]].values         # 將 1 2 3 4 行指定給訓練資料
    y = df.iloc[1:151, 4].values                 # 第 5 行為訓練答案
    y = np.where(y=='Iris-setosa', 0, y)        # 將 Iris-setosa 設為 0   ( 若改為後 100項則註解此行 )
    y = np.where(y=='Iris-virginica', 1, y)     # 將 Iris-virginica 設為 0
    # y = np.where(y=='Iris-versicolor', 0, y)  # ( 若改為後 100項則此註解拿掉 )
    
    x = np.append(x[:50], x[100:], axis=0)      # 將資料前50與後50合併 ( 資料 )  ( 若改為後 100項則註解此行 )
    y = np.append(y[:50], y[100:])              # 將資料前50與後50合併 ( 答案 )  ( 若改為後 100項則註解此行 )
    # x = x[50:]                                # ( 若改為後 100項則此註解拿掉 )
    # y = y[50:]                                # ( 若改為後 100項則此註解拿掉 )
    onehot_y = load_label(y)                    # 將Label onehot編碼

    
    x_train=np.empty((80,4))                    # 訓練 資料
    x_test=np.empty((20,4))                     # 測試 資料
    y_train=np.empty(80)                        # 訓練 ( 答案 )
    y_test=np.empty(20)                         # 測試 ( 答案 )

    x_train[:40], x_train[40:] = x[10:50], x[60:]       # 訓練資料為 10~50 + 60~100
    x_test[:10],  x_test[10:]  = x[:10],   x[50:60]     # 測試資料 0~10 + 50~60
    y_train[:40], y_train[40:] = y[10:50], y[60:]       # 標籤部分(答案)
    y_test[:10],  y_test[10:]  = y[:10],   y[50:60]     # 標籤部分(答案)

    # 輸入層到中間層 權重、偏差
    im_weights = np.ones((4, 2))/10     # (4 x 2) 4為輸入有4個，中間有兩個神經元故為 4x2
    im_bias    = np.ones(2)/10          # 兩個神經元偏差2個

    # 中間層到輸出層 權重、偏差
    mo_weights = np.ones((2, 1))/10     # (2 x 1) 2為輸入有2個來自中間層那兩個神經元的輸出，輸出僅一個神經元故為 2x1
    mo_bias    = np.ones(1)/10          # 一個神經元輸出偏差1個

    # 總共的 權重 以及 偏差
    w_list = [im_weights, mo_weights]   # 將每層權重組成陣列
    b_list = [im_bias, mo_bias]         # 將每層偏差組成陣列

    """
        超參數部分設定
    """
    eta      = 1.0   # 學習率
    epoch    = 1000  # 迭代次數
    showLoss = 100   # 每隔幾次印出

    """
        開始訓練
    """
    for _ in range(1, epoch+1):
        w_list, b_list, loss = update( x_train, y_train, w_list, b_list, eta ) # 開始進行更新
        if _ % showLoss == 0:
            print(f"Epoch is { _ }/{ epoch } times. Loss is { loss }.")        # 印出當前 迭代次數、均方差值

    """
        結束訓練 印出 每層 權重 以及 偏差，若有需要可取消註解
    """
    # print('神經元1', '\n', 'weights=', im_weights[0], '\n', 'bias=',im_bias[0])
    # print('神經元2', '\n', 'weights=', im_weights[1], '\n', 'bias=',im_bias[1])
    # print('輸出元', '\n', 'weights=', mo_weights[1], '\n', 'bias=',mo_bias)

    """
        使用測試資料進行測試
    """
    result = calculate(x_test, w_list, b_list)
    # print("輸入到中間層輸出\n", result['y_1'])
    print("中間到輸出層輸出\n", result['y_2'].reshape(result['y_2'].shape[0]))
    print("正確解答\n", y_test)
