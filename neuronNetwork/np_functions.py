from modules import *
# np.mean vs np.average (Average can have weights)
a = np.array([
    [5, 9, 13],
    [14, 10, 12],
    [11, 15, 19]
])

"""
b = np.average(a)
print(b)
c = np.average(a, axis=0, weights=[1./4, 2./4, 1./4])
print(c)
d = np.average(a, axis=1)
print(d)
"""

x_mean = np.mean(a, axis=0)
x_std = np.std(a, axis=0)
X = (a-x_mean)/x_std
print(X)