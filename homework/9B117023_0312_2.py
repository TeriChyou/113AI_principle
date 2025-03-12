import numpy as np

# Matrix
A = np.array([
        [2, 3, 4, -5],
        [6, 7, -8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17]
    ])

B = np.array([-6, 96, 312, 416])

def solution1():
    det_A = np.linalg.det(A)

    if det_A == 0:
        print("The determinant is zero, so the system has no unique solution.")
    else:
        # 取 A_w
        A_w = A.copy()
        A_w[:, 0] = B
        det_Aw = np.linalg.det(A_w)

        # A_x
        A_x = A.copy()
        A_x[:, 1] = B
        det_Ax = np.linalg.det(A_x)

        # A_y
        A_y = A.copy()
        A_y[:, 2] = B
        det_Ay = np.linalg.det(A_y)

        # A_z
        A_z = A.copy()
        A_z[:, 3] = B
        det_Az = np.linalg.det(A_z)

        # using Cramer's Rule
        w = det_Aw / det_A
        x = det_Ax / det_A
        y = det_Ay / det_A
        z = det_Az / det_A

        print(f"Solution using Cramer's Rule:")
        print(f"Solution: w = {w:.3f}, x = {x:.3f}, y = {y:.3f}, z = {z:.3f}")

def solution2():
    # Inverse of A
    A_inv = np.linalg.inv(A)

    # Solve for X using X = A^(-1) * B
    RES = np.dot(A_inv, B)  # or X = A_inv @ B
    
    w,x,y,z = RES

    print(f"Solution using X = A^(-1)B:")
    print(f"Solution: w = {w:.3f}, x = {x:.3f}, y = {y:.3f}, z = {z:.3f}")

solution1()
solution2()