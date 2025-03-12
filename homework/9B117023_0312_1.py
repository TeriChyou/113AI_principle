import numpy as np
import matplotlib.pyplot as plt

points = np.array([[-1, 1], [1, 2], [3, 4]])
# 先轉換係數 ax^2 + bx + c
A = np.array([[1, -1, 1],
            [1, 1, 1],
            [9, 3, 1]])
# y軸f(x)向量
B = np.array([1, 2, 4])

# Cramer's Rule
def solution1():
    # 計算A的行列式
    det_A = np.linalg.det(A)

    if det_A == 0:
        print("The determinant is zero, so the system has no unique solution.")
    else:
        # 取 A_a
        A_a = A.copy()
        A_a[:, 0] = B
        det_Aa = np.linalg.det(A_a)

        # A_b
        A_b = A.copy()
        A_b[:, 1] = B
        det_Ab = np.linalg.det(A_b)

        # A_c
        A_c = A.copy()
        A_c[:, 2] = B
        det_Ac = np.linalg.det(A_c)

        # using Cramer's Rule
        a = det_Aa / det_A
        b = det_Ab / det_A
        c = det_Ac / det_A

    # 印解
    print(f"Solution using Cramer's Rule:")
    print(f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f}")
    return a,b,c
# AX=b => X = A^(-1)b
def solution2():
    # Inverse of A
    A_inv = np.linalg.inv(A)

    # Solve for X using X = A^(-1) * B
    X = np.dot(A_inv, B)  # or X = A_inv @ B

    # Extract coefficients
    a, b, c = X

    # Print results
    print(f"Solution using X = A^(-1)B:")
    print(f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f}")

def drawPlot(a,b,c):
    # 制定 x,y 範圍
    x_values = np.linspace(-2, 4, 100)  # From -2 to 4
    y_values = a * x_values**2 + b * x_values + c  # Compute y values

    # 將多項式呈現
    plt.plot(x_values, y_values, label=f"${a:.3f}x^2 + {b:.3f}x + {c:.3f}$", color='b')

    #     
    plt.scatter(points[:, 0], points[:, 1], color='r', label="Given Points")
    # Labels and legend
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Quadratic Polynomial Passing Through Given Points")
    plt.legend()
    plt.grid()

    # Show
    plt.show()

a,b,c = solution1()
solution2()
drawPlot(a,b,c)