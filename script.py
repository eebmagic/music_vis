from fft_examples.live_example import get_audio_data, make_fft
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    print('Loading data from file...')
    target_file = 'audio/trimmed_punk_ye.wav'
    sample_rate, data = get_audio_data(target_file)
    x, y = data
    print('Finished loading from file.')

    total_seconds = len(y) / sample_rate
    frame_rate = 30     # fps
    # sample_chunk_width = sample_rate // frame_rate
    tempo = 100     # bpm
    bps = tempo / 60
    samples_per_beat = int(sample_rate / bps)
    # sample_chunk_width = sample_rate
    sample_chunk_width = samples_per_beat
    total_chunks = int(total_seconds * frame_rate)
    frame_step_width = int(sample_rate / frame_rate)

    print(f"\n{total_seconds = }")
    print(f"{total_chunks = }")
    print(f"{frame_rate = }")
    print(f"{bps = }")
    print(f"{samples_per_beat = }")
    print(f"{sample_chunk_width = }")
    print(f"{frame_step_width = }")
    # quit()

    # step = int(total_seconds // frame_rate)
    # starts = list(range(0, len(y)-sample_chunk_width, step))
    # for i in tqdm(range(len(starts)-1)):
    for i in tqdm(range(total_chunks)):
        # start = starts[i]
        start = i * frame_step_width
        stop = start + sample_chunk_width
        if stop > len(y):
            stop = len(y)

        graph, notable = make_fft(y, start, stop, sample_rate=sample_rate)
        if len(notable):
            print(i, notable, len(notable))
        # Wave
        data_segment = y[start:stop]
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(data_segment)) / float(sample_rate), data_segment)
        plt.ylim([-20_000, 20_000])
        plt.title(f"{i}")

        # FFT
        plt.subplot(2, 1, 2)
        plt.plot(graph[0], graph[1])
        plt.ylim([0, 8*1e7])
        plt.xscale('log')

        # plt.show()
        plt.savefig(f'outputs/{i:05}.png')
        plt.clf()