from scipy.io import wavfile
import numpy as np
from numpy.fft import fft, fftfreq

def get_audio_data(filename):
    '''
    PARAMS:
        filename: the file path of the file to get data from

    RETURN:
        (sample_rate, (x_data, y_data))
    '''
    sample_rate, signal = wavfile.read(filename)
    y = signal[:, 0]
    x = np.arange(len(y)) / float(sample_rate)

    return (sample_rate, (x, y))


def make_fft(data, start, stop, sample_rate=44_100, scan_width=1, cutoff=0.5*1e8):
    '''
    PARAMS:
        data: the full data trim/use
        start: the first index to trim at
        stop: the last index to trim at
        sample_rate: the sample rate of the audio file the data is from
        scan_width: the piano note radius to use for making bins for notable freqs
        cutoff: the amplitude over which a frequency is considered notable

    RETURN:
        ((x_data, y_data), list_of_notable_freqs)
    '''
    # Trim data
    data = data[start:stop]

    # Make fft values
    freqs = fftfreq(len(data)) * sample_rate
    mask = np.logical_and(freqs > 0, freqs < 1000)

    fft_vals = fft(data)
    fft_theo = np.abs(fft_vals)

    # Find notable freqs (uses scan_width and cutoff)
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
        if max_key not in real:
            real[max_key] = notable[max_key]
    notable = sorted(list(real))

    fft_graph = (freqs[mask], fft_theo[mask])
    return (fft_graph, notable)


if __name__ == '__main__':
    sample_rate, data = get_audio_data('c_scale_60_bpm.wav')
    x, y = data
    print(sample_rate)
    print(y)
    print(y.shape)
    print(max(y))
    print(min(y))

    import matplotlib.pyplot as plt
    import time
    segment_start = 0
    segment_stop = 14
    for i in range(segment_start, segment_stop+1):
        start_time = time.time()
        start = sample_rate * i
        stop = sample_rate * (i + 1)
        graph, notable = make_fft(y, start, stop, sample_rate=sample_rate)

        print(i, notable, len(notable))

        # Plot wave data for segment
        plt.subplot((segment_stop+1)*2, 1, i*2+1)
        data_segment = y[start:stop]
        plt.plot(np.arange(len(data_segment)) / float(sample_rate), data_segment)

        # Plot fft data for segment
        plt.subplot((segment_stop+1)*2, 1, i*2+2)
        plt.plot(graph[0], graph[1])

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.02, top=0.98, hspace=0.2)
    print('showing graph')
    plt.show()
