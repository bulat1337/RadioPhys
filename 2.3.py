import numpy as np
import matplotlib.pyplot as plt

f0 = 10e3       
tau = 500e-6    

Nt = 10001
t = np.linspace(0, tau, Nt)

x_rect = np.sin(2*np.pi*f0*t)
w_hann = 0.5 + 0.5*np.cos(np.pi*(t - tau/2)/(tau/2))
x_hann = x_rect * w_hann

Nf = 2000
f_band = np.linspace(-2*f0, 2*f0, Nf)

def compute_FT(x, t, f_band):

    E = np.exp(-2j * np.pi * f_band[:, None] * t[None, :])
    X = np.trapz(x[None, :] * E, t, axis=1)
    return X

X_rect = compute_FT(x_rect, t, f_band)
X_hann = compute_FT(x_hann, t, f_band)

idx_f0 = np.argmin(np.abs(f_band - f0))
mag_rect_f0 = np.abs(X_rect[idx_f0])
mag_hann_f0 = np.abs(X_hann[idx_f0])

width_rect = 2.0 / tau
width_hann = 4.0 / tau

plt.figure(figsize=(8,3.5))
plt.plot(np.concatenate((t*1e6, (t+tau)*1e6))[:len(t)], x_rect, label='rect (время)')
plt.plot(t*1e6, x_hann, label='hann (время)', alpha=0.7)
plt.xlabel('t, мкс')
plt.ylabel('x(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(8,3.5))
plt.plot(f_band/1e3, np.abs(X_rect), label='|X| rect')
plt.plot(f_band/1e3, np.abs(X_hann), label='|X| hann', alpha=0.8)
plt.axvline(f0/1e3, color='k', linestyle='--', linewidth=0.6)
plt.axvline(-f0/1e3, color='k', linestyle='--', linewidth=0.6)
plt.xlabel('f, кГц')
plt.ylabel('|X(f)|')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()