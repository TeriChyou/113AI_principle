from modules import *

zeroArr = np.zeros((2,3))

# print(zeroArr)
"""
[[0. 0. 0.]
 [0. 0. 0.]]
"""

d5Arr = np.array([1,2,3,4], ndmin=5)
# print(d5Arr)
# print('Number of dimensions:', d5Arr.ndim)

"""
[[[[[1 2 3 4]]]]]
Number of dimensions: 5
"""

mainArr = np.array([1,3,5,7,9,11,13])
# print(mainArr[1]+mainArr[2])
"""
8 <= proof of array in numpy is zero-indexed
"""
# print(mainArr[1:5])
# print(mainArr[1:5:2])
"""
[3 5 7 9]
[3 7]
"""
main2DArr = np.array([[1,3,5,7,9,11,13], [2,4,6,8,10,12,14]])
# print(main2DArr[1, 1:4])
# print(main2DArr[0:2, 1:4])
"""
[4 6 8]
[[3 5 7]
 [4 6 8]]
"""
arrReshape = np.append(main2DArr[0], main2DArr[1]) # merge 
arrReshape = np.sort(arrReshape) # sort
arrReshape = arrReshape.reshape(2, 7)
#print(arrReshape)
"""
[[ 1  2  3  4  5  6  7]
 [ 8  9 10 11 12 13 14]]
"""