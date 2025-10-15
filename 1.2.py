import os
import wave
import glob
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

audio_dir = "audio"
plot_out_dir = "audio_plots"
os.makedirs(plot_out_dir, exist_ok=True)

windows = [
    (8000, 10000),
    (8000, 10000),
    (8000, 10000),
    (8000, 10000),
    (8000, 10000),
    (8000, 10000)
]

wav_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

for i, path in enumerate(wav_paths):
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)

    start, end = windows[i % len(windows)]

    with wave.open(path, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate  = wf.getframerate()
        nframes    = wf.getnframes()
        comptype   = wf.getcomptype()

    bits_per_sample = sampwidth * 8

    fs, x = wavfile.read(path)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_samples = x.shape[0]
    channels = x.shape[1]

    start = int(start)
    end = int(end)
    if start < 0:
        start = 0
    if end <= start:
        end = min(start + 1, n_samples)
    if start >= n_samples:
        start = 0
        end = min(1, n_samples)
    if end > n_samples:
        end = n_samples

    x_window = x[start:end, 0]
    k_local = np.arange(x_window.size)

    k_window = np.arange(start, end)
    times_window = k_window / fs
    dt_mean = float(np.mean(np.diff(times_window))) if len(times_window) > 1 else None

    duration_seconds = nframes / fs
    estimated_bytes = nframes * channels * (bits_per_sample // 8)
    estimated_kb = estimated_bytes / 1024.0
    actual_bytes = os.path.getsize(path)
    actual_kb = actual_bytes / 1024.0
    ratio = (actual_kb / estimated_kb) if estimated_kb > 0 else float('nan')

    print(f"\nfile: {name}")
    print(f"applied window #{i % len(windows)}: start={start}, end={end} (samples)")
    print(f"fs = {fs} Hz, channels = {channels}, sampwidth = {sampwidth} байт, bits_per_sample = {bits_per_sample}")
    print(f"dtype scipy = {x.dtype}", end="")
    print(f", bits_from_dtype = {x.dtype.itemsize * 8}, levels = {2 ** (x.dtype.itemsize * 8)}")
    print(f"nframes = {nframes}, duration_seconds = {duration_seconds:.6f} s")
    print(f"средний шаг в окне dt_mean = {dt_mean} s, 1/fs = {1.0/fs} s, разница = {None if dt_mean is None else dt_mean - 1.0/fs}")
    print(f"estimated size= {estimated_kb:.2f} kb, actual_kb = {actual_kb:.2f} kb, ratio = {ratio:.3f}")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(k_local / fs, x_window, '.')
    plt.title(f"{base}: k_local/fs")
    plt.xlabel("t, s")
    plt.grid(True)

    plt.subplot(1,2,2)
    t_correct = (start + k_local) / fs
    plt.plot(t_correct, x_window, '.')
    plt.title(f"{base}: время от начала файла")
    plt.xlabel("t, s")
    plt.grid(True)

    out_png = os.path.join(plot_out_dir, f"{base}_analysis.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
