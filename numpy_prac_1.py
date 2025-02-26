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
main2DArr = np.array([[1,3,5,7,9,11,13,15], [2,4,6,8,10,12,14,16]])
# print(main2DArr[1, 1:4])
# print(main2DArr[0:2, 1:4])
"""
[4 6 8]
[[3 5 7]
 [4 6 8]]
"""
arrReshape = np.append(main2DArr[0], main2DArr[1]) # merge 
arrReshape = np.sort(arrReshape) # sort
arrReshape_1 = arrReshape.reshape(2, 8)
arrReshape_2 = arrReshape.reshape(2, 2, 4)
# print(arrReshape_1)
"""
[[ 1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16]]
"""
# print(arrReshape_2)
"""
[[[ 1  2  3  4]
  [ 5  6  7  8]]

 [[ 9 10 11 12]
  [13 14 15 16]]]
"""
for x in arrReshape_2:
    for y in x:
        for z in y:
            # print(z, end=",")
            break
for x in np.nditer(arrReshape_2):
    # print(x, end=",")
    break
"""
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  => Same output
"""
for x in np.nditer(arrReshape_1[:,::2]):
    # print(x, end=",")
    break
"""
1,3,5,7,9,11,13,15, 
=> array[row_start:row_end:row_step, column_start:column_end:column_step]
The first part : means "select all rows".
The second part ::2 means "select every second column, starting from the first column (index 0)". 
"""
for index, x in np.ndenumerate(arrReshape_1):
    # print(f"{index}:{x}")
    break
"""
(0, 0):1
(0, 1):2
(0, 2):3
(0, 3):4
(0, 4):5
(0, 5):6
(0, 6):7
(0, 7):8
(1, 0):9
(1, 1):10
(1, 2):11
(1, 3):12
(1, 4):13
(1, 5):14
(1, 6):15
(1, 7):16
"""

arrOne = np.array([[1,2,3,4],[5,6,7,8]])
arrTwo = np.array([[9,10,11,12],[13,14,15,16]])

arrAxisZero = np.concatenate((arrOne, arrTwo), axis=0)
arrAxisOne = np.concatenate((arrOne, arrTwo), axis=1)
# print(f"{arrAxisZero}\n{arrAxisOne}")
"""
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

[[ 1  2  3  4  9 10 11 12]
 [ 5  6  7  8 13 14 15 16]]
"""
# print(np.hstack((arrOne, arrTwo))) # horizontal stack
# print(np.vstack((arrOne, arrTwo))) # vertical stack
# print(np.dstack((arrOne, arrTwo))) # depth stack
"""
# Horizontal
[[ 1  2  3  4  9 10 11 12]
 [ 5  6  7  8 13 14 15 16]]
# Vertical
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]
# Depth
 [[[ 1  9]
  [ 2 10]
  [ 3 11]
  [ 4 12]]

 [[ 5 13]
  [ 6 14]
  [ 7 15]
  [ 8 16]]]
"""
# print(np.array_split(arrReshape, 4)) each in One Dim
"""
[array([1, 2, 3, 4]), array([5, 6, 7, 8]), array([ 9, 10, 11, 12]), array([13, 14, 15, 16])]
"""
# print(np.array_split(np.vstack((arrOne, arrTwo)), 4)) each in Two Dim
"""
[array([[1, 2, 3, 4]]), array([[5, 6, 7, 8]]), array([[ 9, 10, 11, 12]]), array([[13, 14, 15, 16]])]
"""

filterArr = np.array([1,2,3,4,5,6,7,8,9,10])
filterGoblin = [True, False, True, False, True, False, True, False, True, False]
# print(filterArr[filterGoblin])
"""
[1 3 5 7 9]
"""
# So, we can manipulate to filter out the data that we don't want.
filterGoblin2 = []
for x in filterArr:
    if x%2 == 0:
        #filterGoblin2.append(True)
        break
    else:
        #filterGoblin2.append(False)
        break
# print(filterArr[filterGoblin2]) => Will only return even numbers.
"""
[ 2  4  6  8 10]
"""

# RANDOM

# x = np.random.randint(100, size=(5))
# y = np.random.rand(5)
# print(x, y)
"""
[ 3  0 75 26 52] [0.17227353 0.79798807 0.73216065 0.27011925 0.66104404]
"""
# x = np.random.randint(100, size=(3,5))
# y = np.random.rand(3,5)
# print(x, y)
"""
[[ 9 39 75 71 41]
 [ 5 84 44 79  2]
 [58  9 68 93 75]]

 [[0.84671759 0.84678452 0.04647903 0.71079082 0.27253466]
 [0.96871232 0.29437179 0.67458943 0.19390783 0.97867208]
 [0.18559731 0.61419746 0.97247279 0.63728054 0.67477665]]
"""
# x = np.random.choice([1,3,5,7]) => Namely
x = np.array([1,3,5,7,9,11,13])
# print(np.random.choice(x, size=(3,5))) => Randomly choose num in arr then put into 3x5 array
"""
[[ 1  7  3 13 11]
 [ 3 13  5  5  1]
 [11  5 11  7  5]]
"""