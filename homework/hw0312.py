import numpy as np
import matplotlib.pyplot as plt
# AX=b => X = A^(-1)b

def question1():
    points = np.array([[-1, 1], [1, 2], [3, 4]])
    # 先轉換係數 ax^2 + bx + c
    A = np.array([[1, -1, 1],
                [1, 1, 1],
                [9, 3, 1]])

    # y軸f(x)向量
    B = np.array([1, 2, 4])

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
    print(f"Solution: a = {a:.3f}, b = {b:.3f}, c = {c:.3f}")

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

def question2():
    # Make the matrix
    A = np.array([
        [2, 3, 4, -5],
        [6, 7, -8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17]
    ])

    B = np.array([-6, 96, 312, 416])

    res = np.linalg.solve(A,B)
    w,x,y,z = res
    print(f"Solution: w = {w:.3f}, x = {x:.3f}, y = {y:.3f}, z = {z:.3f}")

question2()
