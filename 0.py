import numpy as np
import matplotlib.pyplot as plt

arr_a = np.array([1, 3, 5, 7, 9])
arr_b = np.arange(1, 10, 2)
arr_c = np.linspace(1, 9, 5, dtype=int)

k = np.arange(100)
ratio = 0.07
x = np.sin(2 * np.pi * ratio * k)

plt.figure(figsize=(12, 5))
plt.plot(k, x)
plt.xlabel('k')
plt.ylabel('x[k]')
plt.title('x[k] = sin(2π * 0.07 * k)')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(k, x, 'r--', label='r-- (пунктир, красный)')
plt.stem(k, x, basefmt=" ", label='отсчеты (stem)')
plt.xlabel('k')
plt.ylabel('x[k]')
plt.title('x[k] — пунктирная линия и отсчеты (stem)')
plt.legend()
plt.grid(True)
plt.show()

def my_fun(x):
    x = np.asarray(x)
    return np.where(x == 0, 1.0, np.sin(x) / x)

y = my_fun(2 * np.pi * ratio * k)

plt.figure(figsize=(12, 5))
plt.plot(k, y, 'b-')
plt.xlabel('k')
plt.ylabel('y[k] = sin(θ)/θ, θ=2π·0.07·k')
plt.title('y[k] = sinc-like функция my_fun(2π·0.07·k)')
plt.grid(True)
plt.show()

z = np.exp(-1j * 2 * np.pi * ratio * k)

plt.figure(figsize=(12, 5))
plt.plot(k, z.real, label='Re{z[k]}')
plt.xlabel('k')
plt.ylabel('Re{z[k]}')
plt.title('Реальная часть z[k] = exp(-j·2π·0.07·k)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(k, z.imag, label='Im{z[k]}')
plt.xlabel('k')
plt.ylabel('Im{z[k]}')
plt.title('Мнимая часть z[k] = exp(-j·2π·0.07·k)')
plt.grid(True)
plt.legend()
plt.show()