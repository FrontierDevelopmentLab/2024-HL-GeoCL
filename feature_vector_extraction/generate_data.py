import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = np.sin(2 * np.pi * freq * t)
    return t, y

if __name__ == "__main__":
    times, data = generate_sine_wave(freq=1, sample_rate=100, duration=10)  # 1 Hz sine wave
    plt.plot(times, data)
    plt.title("Synthetic Sine Wave")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
