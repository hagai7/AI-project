import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write
import os.path

DISPLAY_PLOTS = False
USAGE_MSG = f"Usage: python3 {os.path.basename(__file__)} input_wav_path output_wav_path"
DEFAULT_OUTPUT_WAV_NAME = 'polyfit_result.wav'


"""polynomial fitting function"""
def polyfit(audio):
    x = np.arange(0, len(audio))
    mymodel = np.poly1d(np.polyfit(x, audio, 50))

    if DISPLAY_PLOTS:
        plt.scatter(x, audio, label="Positive data")
        plt.plot(x, mymodel(x), color="orange", label="Polynomial")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
    return mymodel(x)


"""Input validation"""
if len(sys.argv) != 3:
    print(USAGE_MSG)
    exit()

input_path, output_path = sys.argv[1], sys.argv[2]

if not os.path.exists(input_path) or not os.path.isfile(input_path):
    print('Invalid input path')
    exit()

if os.path.exists(output_path) and os.path.isdir(output_path):
    output_path = os.path.join(output_path, DEFAULT_OUTPUT_WAV_NAME)

"""read user input (wav file)"""
samplerate, data = wavfile.read(input_path)
data = data.copy()
length = data.shape[0]
if length == 0:
    print("Input is invalid")
    exit()
if len(data.shape) > 1:
    data = data[:, 1]

time = np.arange(0, length)

if DISPLAY_PLOTS:
    plt.plot(time, data, label="Original data")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

"""changing negative values to positive"""
pos_neg = [1]*len(data)
avg = np.average(data)
for i in range(len(data)):
    if data[i] < avg:
        data[i] = 2*avg - data[i]
        pos_neg[i] = 0

if DISPLAY_PLOTS:
    plt.plot(time, data, label="Positive data")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

"""polynomial fitting"""
poly_res = polyfit(data)
poly_res *= 4

if DISPLAY_PLOTS:
    plt.plot(time, poly_res, label="Polynomial - positive and multiplied by constant")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


"""return positive elements to negative"""
avg = np.average(data)
for i in range(len(poly_res)):
    if pos_neg[i] == 0:
        poly_res[i] = 2*avg - poly_res[i]

if DISPLAY_PLOTS:
    plt.plot(time, poly_res, label="Polynomial - final")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

"""extract final result to wav file"""
write(output_path, 44000, poly_res.astype(np.dtype('i2')))
