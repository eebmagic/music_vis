'''
fft example with loaded sound data
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq

from scipy.io import wavfile

# Load data
sample_rate, signal = wavfile.read("c_scale_60_bpm.wav")
# start = sample_rate * 7
# stop = sample_rate * 8
start = sample_rate * 0
stop = sample_rate * 15
y = signal[start:stop, 0]
x = np.arange(len(y)) / float(sample_rate)

# Plot waves
plt.subplot(2, 1, 1)
plt.plot(x, y, label='y')

# Make mask (used to avoid negative freq mirroring in output)
freqs = fftfreq(len(y)) * sample_rate
mask = np.logical_and(freqs > 0, freqs < 1000)

# Make fft data (and normalized data)
fft_vals = fft(y)
fft_theo = np.abs(fft_vals)

# Find the loudest freqs just for fun
scan_width = 1
cutoff = 0.5 * 1e8

loudest_freq = abs(freqs[np.argmax(fft_vals)])

notable = {}
r = 2 ** (1 / 12)
for x, y in zip(freqs[mask], fft_theo[mask]):
    x = int(x)
    if y > cutoff:
        notable[x] = y

real = {}
for x in notable:
    low = int(x * (r ** -scan_width))
    high = int(x * (r ** scan_width))
    subset = {x: notable[x]}
    for x2 in range(low, high):
        if x2 in notable:
            subset[x2] = notable[x2]
    max_key = max(subset, key=subset.get)
    # print(f'choosing {max_key} from {subset.keys()}')
    real[max_key] = notable[max_key]
notable = sorted(list(real))

print('\n')
print("notable freqs:", notable, len(notable))
print("loudest freq:", loudest_freq)


# Plot fft results
plt.subplot(2, 1, 2)
plt.plot(freqs[mask], fft_theo[mask], label='theo')
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.legend()
plt.title('fft results')
plt.show()