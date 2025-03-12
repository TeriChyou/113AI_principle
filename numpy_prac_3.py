from modules import *
def show():
    mpl.pyplot.show()
# Method wihout ufunc => UniversalFunction
#x = [1,2,3,4]
#y = [5,6,7,8]
#z = []

"""
for i, j in zip(x,y):
    z.append(i + j)
print(z)
"""
# just like
# z = np.add(x,y)


# arr1 = np.array([10,12,14,15,16,17])
# arr2 = np.array([4,5,61,23,42,5])
"""
res = np.subtract(arr1, arr2)
print(res)

res = np.multiply(arr1,arr2)
print(res)

res = np.dot(arr1,arr2)
print(res)

res = np.divide(arr1,arr2)
print(res)

res = np.mod(arr1,arr2)
print(res)
# eqv ↕
res = np.remainder(arr1,arr2)
print(res)

res = np.divmod(arr1,arr2)
print(res)
"""
# axis practice
"""
arr1 = np.array([[1,2,3], [4,5,6]])
arr2 = np.array([[1,2,3], [4,5,6]])

for i in range(0,3):
    res = np.sum([arr1,arr2], axis=i)
    print(res)

"""
# product
"""
arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
x = np.prod(arr1)
y = np.prod([arr1, arr2])
print(f"{x}, {y}")
"""

# sinValue = np.sin(np.pi/2)
# print(sinValue)
# Show sine wave 
"""
import matplotlib.pyplot as plt

# Define x values (0 to 100)
x = np.linspace(0, 360, 10)  # More points for a smooth curve

# Define y values as the sine wave
y = np.sin(x * (np.pi / 180))  # Adjust frequency using π/100

# Plot the sine wave
plt.plot(x, y, label="Sine Wave", linewidth=2, color='b')

# Labels and title
plt.xlabel("x values")
plt.ylabel("sin(x * π / 100)")
plt.title("Sine Wave Plot")

# Show legend
plt.legend()

# Show the plot
plt.show()
"""

# Axis practice 2
"""
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
"""
# Discrete Difference (差分)
"""
arr = np.array([10,15,25,5])
newArr = np.diff(arr) # List of => arr[i+1] - arr[i]
print(newArr)
"""

# Derivative
def differentiate():
    x = np.array([4, 3, 4, 5])
    dx = np.polyder(x)
    print(dx)

# differentiate()