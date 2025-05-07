from modules import *

x_0 = np.arange(-1.0, 1.0, 0.1)
x_1 = np.arange(-1.0, 1.0, 0.1)
Z = np.zeros((400, 3)) # n = 3

w_im = np.array([
    [1.0, 2.0],
    [2.0, 3.0]
])

w_mo = np.array([
    [-1.0, 1.0, 1.0],
    [1.0, -1.0, 2.0]
])

b_im = np.array([0.3, -0.3])
b_mo = np.array([0.4, 0.1, 0.2])

def middle_layer(x, w, b):

  u = np.dot(x, w) + b
  # 應用 sigmoid activation function
  # return 1/(1+np.exp(-u))
  # change activation function
  return np.where(u <= 0, 0.01*u, u) # or np.tanh(u) => whatevery function

def output_layer(x, w, b):
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

print(Z) 

plus_x = []
plus_y = []
circle_x = []
circle_y = []

for i in range(20):
  for j in range(20):
    if Z[i*20 + j][0] > Z[i*20+j][0]:
      plus_x.append(x_0[i])
      plus_y.append(x_1[j])
    else:
      circle_x.append(x_0[i])
      circle_y.append(x_1[j])
plt.scatter(plus_x, plus_y, marker="o")
plt.scatter(circle_x, circle_y, marker="x")
plt.show()