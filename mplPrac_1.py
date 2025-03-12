from modules import *
x = np.linspace(0, 10, 100)  # 100 evenly spaced values between 0 and 10
y = np.sin(x)

plt.plot(x, y, label="Sine Wave", color="b", linestyle="--", linewidth=2)
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Basic Line Plot")
plt.legend()
plt.show()
