import numpy as np
import matplotlib.pyplot as plt

tau = 200e-6               

T = 1.05 * tau             

t_plot = np.linspace(-0.1*tau, 2.0*tau, 1200)

def X_rect_mag(f, T):
    
    out = np.zeros_like(f, dtype=float)
    nz = np.abs(f) > 1e-30
    out[nz] = np.abs(np.sin(np.pi * f[nz] * T) / (np.pi * f[nz]))
    out[~nz] = T
    return out
    
def rect_t(t, T):
    return np.where((t >= 0.0) & (t <= T), 1.0, 0.0)
    
def sinc_reconstruct(t, tk, xk):
    
    Ts = tk[1] - tk[0]
    
    A = np.sinc((t[:, None] - tk[None, :]) / Ts)
    return A.dot(xk)
    
def plot_case(fs, ax_time, ax_freq, plot_title):
    Ts = 1.0 / fs
    
    t_max = 2 * tau
    tk = np.arange(0.0, t_max + Ts/2, Ts)
    xk = rect_t(tk, T)                
    


    t_rec = np.linspace(-0.05*tau, 1.5*tau, 800)
    x_rec = sinc_reconstruct(t_rec, tk, xk)


    f = np.linspace(-1.5*fs, 1.5*fs, 2000)
    Xc = X_rect_mag(f, T)


    n_max = 6
    n_range = np.arange(-n_max, n_max+1)
    X_periodic = np.zeros_like(f)
    for n in n_range:
        X_periodic += X_rect_mag(f - n*fs, T)
    X_periodic *= 1.0 / Ts


    M = 65536
    Xk = np.fft.fftshift(np.fft.fft(xk, M))
    f_fft = np.fft.fftshift(np.fft.fftfreq(M, d=Ts))
    Xk_mag = np.abs(Xk) * (1.0 / Ts) 
    


    ax_time.plot(t_plot*1e6, rect_t(t_plot, T), color='g', label='x(t) — непрерывный прямоугольник')
    ax_time.stem(tk*1e6, xk, linefmt='b-', markerfmt='bo', basefmt=' ', label='отсчёты x[k]')
    ax_time.plot(t_rec*1e6, x_rec, color='c', linewidth=1.2, label='восстановление sinc (Котельников)')
    ax_time.set_xlabel('Время, мкс')
    ax_time.set_ylabel('Амплитуда')
    ax_time.set_title(plot_title + ' — временная область')
    ax_time.grid(True)
    ax_time.legend(loc='best', fontsize='small')


    ax_freq.plot(f/1e3, Xc, color='g', label='|X_c(f)| — непр. спектр (аналитич.)')
    ax_freq.plot(f/1e3, X_periodic, color='r', label='(1/Ts)·∑ X_c(f - n f_s) — после выборки (копии)')
    
    from numpy import interp
    
    mask = (f_fft >= f.min()) & (f_fft <= f.max())
    if np.any(mask):
        Xk_seg = Xk_mag[mask]
        fseg = f_fft[mask]
        Xk_on_f = interp(f, fseg, Xk_seg)


        if np.nanmax(Xk_on_f) > 0:
            Xk_on_f *= (np.nanmax(X_periodic) / np.nanmax(Xk_on_f))
        ax_freq.plot(f/1e3, Xk_on_f, 'b--', label='FFT от отсчётов (интерп.)')

    ax_freq.set_xlabel('Частота, кГц')
    ax_freq.set_ylabel('Магнитуда (отн.)')
    ax_freq.set_title(plot_title + ' — частотная область')
    ax_freq.grid(True)
    ax_freq.legend(loc='best', fontsize='small')
    
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fs1 = 10.0 / tau 

fs2 = 2.0 / tau  


plot_case(fs1, axes[0,0], axes[0,1], f'f_s = 10/τ = {fs1:.0f} Hz')
plot_case(fs2, axes[1,0], axes[1,1], f'f_s = 2/τ = {fs2:.0f} Hz')

plt.tight_layout()
plt.show()
