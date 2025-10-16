import numpy as np
import matplotlib.pyplot as plt

E = 0.1        
tau = 150e-6   

f_band = np.linspace(-4.0/tau, 4.0/tau, 1000) 

def sinc(x):
    z = np.zeros_like(x, dtype=float)
    small = np.abs(x) < 1e-16
    z[small] = 1.0
    z[~small] = np.sin(x[~small]) / x[~small]
    return z

X_common = E * tau * sinc(np.pi * f_band * tau)

X_center = X_common.copy()  

X_start0 = X_common * np.exp(-1j * np.pi * f_band * tau)

abs_center = np.abs(X_center)
abs_start0 = np.abs(X_start0)
re_center = np.real(X_center)
re_start0 = np.real(X_start0)
im_center = np.imag(X_center)
im_start0 = np.imag(X_start0)

max_abs_diff = np.max(np.abs(abs_center - abs_start0))

eps = 1e-20
mask = np.abs(X_center) > eps 
ratio = X_start0[mask] / X_center[mask]
theoretical = np.exp(-1j * np.pi * f_band[mask] * tau)
max_ratio_err = np.max(np.abs(ratio - theoretical))

phase_ratio = np.angle(ratio)               
phase_theor = np.angle(theoretical)
phase_ratio_un = np.unwrap(phase_ratio)
phase_theor_un = np.unwrap(phase_theor)
max_phase_diff = np.max(np.abs(phase_ratio_un - phase_theor_un))

plt.figure(figsize=(9,4))
plt.plot(f_band/1e3, abs_center, label='|X(f)|, центрированный (t∈[-τ/2,τ/2])')
plt.plot(f_band/1e3, abs_start0, '--', label='|X(f)|, начинается в t=0 (t∈[0,τ])')
plt.xlabel('f, кГц')
plt.ylabel('|X(f)|, В·с')
plt.title('Модуль спектра |X(f)| — сравнение (центр / старт в 0)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(9,4))
plt.plot(f_band/1e3, re_center, label='Re X (центрированный)')
plt.plot(f_band/1e3, re_start0, '--', label='Re X (начинается в 0)')
plt.xlabel('f, кГц')
plt.ylabel('Re X(f), В·с')
plt.title('Действительная часть спектра Re X(f)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(9,4))
plt.plot(f_band/1e3, im_center, label='Im X (центрированный)')
plt.plot(f_band/1e3, im_start0, '--', label='Im X (начинается в 0)')
plt.xlabel('f, кГц')
plt.ylabel('Im X(f), В·с')
plt.title('Мнимая часть спектра Im X(f)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(9,4))
plt.plot(f_band[mask]/1e3, phase_ratio_un, label='arg(X_start0/X_center)')
plt.plot(f_band[mask]/1e3, phase_theor_un, '--', label='arg(exp(-jπ f τ))')
plt.xlabel('f, кГц')
plt.ylabel('Фаза, рад')
plt.title('Фазовая проверка: arg(X_start0/X_center) vs -π f τ')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.show()
