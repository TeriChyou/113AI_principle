from modules import *


def differentiate():
    x = np.array([4, 3, 4, 5])
    dx = np.polyder(x)
    print(dx)

differentiate()