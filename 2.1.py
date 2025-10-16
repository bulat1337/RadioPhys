import numpy as np
import matplotlib.pyplot as plt

tau = 150e-6
f = np.linspace(-8/tau, 8/tau, 2000)

sinc = lambda x: np.where(x==0.0, 1.0, np.sin(x)/x)

Xb = tau * sinc(np.pi * f * tau)                      # прямоугольное
Xt = (tau/2) * (sinc(np.pi * f * tau / 2))**2         # треугольное
Xh = (tau/2)*sinc(np.pi*f*tau) + (tau/4)*(sinc(np.pi*(f*tau-1)) + sinc(np.pi*(f*tau+1)))  # хан

def to_dB(X):
    X0 = np.abs(X[f==0])[0] if np.any(f==0) else np.abs(X).max()
    return 20*np.log10(np.abs(X)/X0 + 1e-20)

plt.figure()
plt.plot(f/1e3, np.abs(Xb), label='rect |X|')
plt.plot(f/1e3, np.abs(Xt), label='tri |X|')
plt.plot(f/1e3, np.abs(Xh), label='hann |X|')
plt.xlim(-40, 40)  # кГц
plt.xlabel('f, kHz'); plt.ylabel('|X(f)|'); plt.legend(); plt.grid()

plt.figure()
plt.plot(f/1e3, to_dB(Xb), label='rect dB')
plt.plot(f/1e3, to_dB(Xt), label='tri dB')
plt.plot(f/1e3, to_dB(Xh), label='hann dB')
plt.xlim(0, 60); plt.ylim(-80, 5)
plt.xlabel('f, kHz'); plt.ylabel('20 log10(|X/X(0)|), dB'); plt.legend(); plt.grid()
plt.show()
