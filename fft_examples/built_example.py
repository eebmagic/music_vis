'''
fft example with artificial built data
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq

# Set params
n = 1_000
L = 100
omg = 2 * np.pi / L

# Make waves
x = np.linspace(0, L, n)
y1 = 1.0 * np.cos(5.0 * omg * x)
y2 = 2.0 * np.sin(10.0 * omg * x)
y3 = 0.5 * np.sin(20.0 * omg * x)
y = y1 + y2 + y3

# Plot waves
plt.subplot(3, 1, 1)
plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')
plt.plot(x, y3, label='y3')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(x, y, label='y')

# Make mask (used to avoid negative freq mirroring in output)
freqs = fftfreq(n)
# mask = freqs > 0
mask = np.logical_and(freqs > 0, freqs < 0.04)

# Make fft data (and normalized data)
fft_vals = fft(y)
fft_theo = 2.0 * np.abs(fft_vals / n)

# Plot fft results
plt.subplot(3, 1, 3)
# plt.plot(freqs[mask], fft_vals[mask], label='vals')
plt.plot(freqs[mask] * 1000, fft_theo[mask], label='theo')
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.legend()
plt.title('fft results')
plt.show()