import numpy as np
import matplotlib.pyplot as plt

N = 3
tau = 100e-6
period = 400e-6
A = 0.1 

f_band = np.linspace(-4/tau, 4/tau, 2000)

dt = tau / 250.0
t = np.arange(0.0, N * period, dt)

x = np.zeros_like(t)
for n in range(N):
    start = n * period
    stop = start + tau
    x += A * ((t >= start) & (t < stop))

exp_mat = np.exp(-2j * np.pi * np.outer(f_band, t))
Xnum = exp_mat.dot(x) * dt

sinc_part = np.abs(np.sinc(f_band * tau))
den = np.sin(np.pi * f_band * period)
num = np.sin(np.pi * f_band * N * period)

ratio = np.where(np.isclose(den, 0.0, atol=1e-16), N, np.abs(num / den))

Xanal_mag = A * tau * sinc_part * ratio

Xnum_mag = np.abs(Xnum)

eps = 1e-20
rel_err = np.max(np.abs(Xnum_mag - Xanal_mag) / np.maximum(Xanal_mag, eps))

plt.figure(figsize=(8, 3.5))
plt.plot(t * 1e6, x)
plt.xlabel("Время $t$, мкс")
plt.ylabel("$x(t)$, В")
plt.title(f"Временный сигнал: N={N}, τ={tau*1e6:.0f} мкс, T={period*1e6:.0f} мкс")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3.5))
plt.plot(f_band / 1e3, Xnum_mag, label="Численный |X(f)|")
plt.plot(f_band / 1e3, Xanal_mag, label="Аналитический |X(f)|", linestyle="--")
plt.xlabel("Частота $f$, кГц")
plt.ylabel("$|X(f)|$, В·c")
plt.title("Сравнение спектров (линейная шкала)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
