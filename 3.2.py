

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

K1 = 40
K2 = 80

fs_orig, x = wavfile.read('tuning-fork.wav')  
if x.ndim == 2:
    x = x.mean(axis=1)
if x.dtype == np.int16:
    x = x.astype(np.float32) / 32768.0
elif x.dtype == np.int32:
    x = x.astype(np.float32) / 2147483648.0
elif x.dtype == np.uint8:
    x = (x.astype(np.float32) - 128) / 128.0
else:
    x = x.astype(np.float32)

def spectrum(x_sig, fs, nfft=2**16):
    X = np.fft.fftshift(np.fft.fft(x_sig, nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0/fs))
    mag = np.abs(X) / nfft
    return f, mag

N_view = min(len(x), fs_orig) 
t = np.arange(N_view) / fs_orig

y1 = x[::K1]
y2 = x[::K2]
fs1 = fs_orig / K1
fs2 = fs_orig / K2

nfft = 2**17
f0, X0 = spectrum(x, fs_orig, nfft=nfft)
f1, X1 = spectrum(y1, fs1, nfft=nfft)
f2, X2 = spectrum(y2, fs2, nfft=nfft)

f_tone = 440.0 
nyq_orig = fs_orig / 2.0
nyq1 = fs1 / 2.0
nyq2 = fs2 / 2.0

def aliased_freq(f, fs_new):
    n = int(np.round(f / fs_new))
    f_a = abs(f - n * fs_new)

    if f_a > fs_new / 2:
        f_a = abs(f_a - fs_new)
    return f_a, n

f_a1, n1 = aliased_freq(f_tone, fs1)
f_a2, n2 = aliased_freq(f_tone, fs2)

plt.figure(figsize=(10, 7))

ax1 = plt.subplot(3, 1, 1)
ax1.plot(f0, X0, linewidth=0.6)
ax1.set_xlim(-2000, 2000)     
ax1.set_title("Спектр исходного сигнала (fs = {:.0f} Hz)".format(fs_orig))
ax1.set_xlabel("Частота, Гц")
ax1.set_ylabel("|X(f)|")
ax1.grid(True)

ax2 = plt.subplot(3, 1, 2)
ax2.plot(f1, X1, linewidth=0.6)
ax2.set_xlim(-500, 500)       
ax2.set_title(f"Спектр прореженного сигнала (K1={K1}, fs1={fs1:.3f} Hz)")
ax2.set_xlabel("Частота, Гц")
ax2.set_ylabel("|Y1(f)|")
ax2.axvline(f_a1, color='r', linestyle='--', label=f"теорет. алиас {f_a1:.2f} Hz, n={n1}")
ax2.axvline(-f_a1, color='r', linestyle='--')
ax2.legend()
ax2.grid(True)

ax3 = plt.subplot(3, 1, 3)
ax3.plot(f2, X2, linewidth=0.6)
ax3.set_xlim(-500, 500)
ax3.set_title(f"Спектр прореженного сигнала (K2={K2}, fs2={fs2:.3f} Hz)")
ax3.set_xlabel("Частота, Гц")
ax3.set_ylabel("|Y2(f)|")
ax3.axvline(f_a2, color='r', linestyle='--', label=f"теорет. алиас {f_a2:.2f} Hz, n={n2}")
ax3.axvline(-f_a2, color='r', linestyle='--')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
