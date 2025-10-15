import os
import csv
import numpy as np
import matplotlib.pyplot as plt

variants = {
    1:  {"N": 15, "f0": 200.0,  "fs":  500.0},
    2:  {"N": 20, "f0": 250.0,  "fs": 1250.0},
    3:  {"N": 30, "f0": 400.0,  "fs": 2000.0},
    4:  {"N": 12, "f0": 200.0,  "fs":  600.0},
    5:  {"N": 15, "f0": 600.0,  "fs": 1500.0},
    6:  {"N": 30, "f0": 750.0,  "fs": 3750.0},
    7:  {"N": 25, "f0":1200.0,  "fs": 6000.0},
    8:  {"N": 18, "f0": 600.0,  "fs": 1800.0},
    9:  {"N": 10, "f0": 400.0,  "fs": 1000.0},
    10: {"N": 25, "f0": 500.0,  "fs": 2500.0},
    11: {"N": 20, "f0": 800.0,  "fs": 4000.0},
    12: {"N": 15, "f0": 400.0,  "fs": 1200.0},
}

NUM_LEVELS = 8
QUANT_MIN = -1.0
QUANT_MAX =  1.0

def quantize_uniform(x, quant_min, quant_max, num_levels):
    x = np.asarray(x)
    Delta = (quant_max - quant_min) / num_levels
    idx = np.floor((x - quant_min) / Delta).astype(int)
    idx = np.clip(idx, 0, num_levels - 1)
    levels = quant_min + (idx + 0.5) * Delta
    return levels

out_dir = "results_variants"
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(out_dir, "quantization_summary.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["variant", "N", "f0_Hz", "fs_Hz", "num_levels", "Delta", "theory_eps_max", "empirical_eps_max"])

    for vnum, params in variants.items():
        N = int(params["N"])
        f0 = float(params["f0"])
        fs = float(params["fs"])

        k = np.arange(N)
        t_samples = k / fs
        t_cont = np.linspace(0, (N - 1) / fs, 1024)

        x_samples = np.sin(2 * np.pi * (f0 / fs) * k)   # x[k]
        x_cont = np.sin(2 * np.pi * f0 * t_cont)        # x(t)
        y_quant = quantize_uniform(x_samples, QUANT_MIN, QUANT_MAX, NUM_LEVELS)
        abs_err = np.abs(x_samples - y_quant)

        Delta = (QUANT_MAX - QUANT_MIN) / NUM_LEVELS
        theory_eps_max = Delta / 2.0
        empirical_eps_max = float(np.max(abs_err))

        writer.writerow([vnum, N, f0, fs, NUM_LEVELS, Delta, theory_eps_max, empirical_eps_max])

        # аналоговый сигнал + отсчеты
        fig1, ax1 = plt.subplots(figsize=(8,3))
        ax1.plot(t_cont, x_cont, label='аналоговый x(t)')
        ax1.stem(t_samples, x_samples, label='отсчеты x[k]', basefmt=" ")
        ax1.set_xlabel('t, с')
        ax1.set_ylabel('Амплитуда')
        ax1.set_title(f'Вариант {vnum}: аналоговый и отсчёты (N={N}, f0={f0} Hz, fs={fs} Hz)')
        ax1.grid(True)
        ax1.legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(os.path.join(out_dir, f"variant_{vnum:02d}_samples.png"))
        plt.close(fig1)

        # аналоговый сигнал + квантованный
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(t_cont, x_cont, label='аналоговый x(t)')
        ax2.stem(t_samples, y_quant, label=f'квантованный y[k] (L={NUM_LEVELS})', basefmt=" ")
        if NUM_LEVELS < 21:
            ticks = QUANT_MIN + (np.arange(NUM_LEVELS) + 0.5) * Delta
            ax2.set_yticks(ticks)
        ax2.set_xlabel('t, с')
        ax2.set_ylabel('Амплитуда')
        ax2.set_title(f'Вариант {vnum}: квантование')
        ax2.grid(True)
        ax2.legend(loc='best')
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"variant_{vnum:02d}_quantized.png"))
        plt.close(fig2)

        # ошибка квантования
        fig3, ax3 = plt.subplots(figsize=(8,3))
        ax3.stem(t_samples, abs_err, label='|x[k]-y[k]|', basefmt=" ")
        ax3.set_xlabel('t, с')
        ax3.set_ylabel('Абсолютная ошибка')
        ax3.set_title(f'Вариант {vnum}: ошибка квантования (max={empirical_eps_max:.6f})')
        ax3.grid(True)
        ax3.legend(loc='best')
        fig3.tight_layout()
        fig3.savefig(os.path.join(out_dir, f"variant_{vnum:02d}_error.png"))
        plt.close(fig3)
