from modules import *

arr1 = np.array([[ 1,  2],
                 [ 3,  4],
                 [ 5,  6]])

arr2 = np.array([[ 7,  8],
                 [ 9, 10],
                 [11, 12]])

arr3 = np.array([[13, 14],
                 [15, 16],
                 [17, 18]])

for i in range(0,3):
    res = np.sum([arr1,arr2, arr3], axis=i)
    print(f"Axis{i}'s sum : {res}")