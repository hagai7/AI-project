"""
A collection of fitness functions to choose from,
    each expecting the original audio's data and sample rate, and the tune to compare it with.
    They return a float between [0, 1] representing fitness, similarity.
"""

import librosa
import numpy as np

from configuration import FitnessFunctionsConf


def zcr_rms_distance(original_rms: np.ndarray, original_zcr: np.ndarray, tune) -> float:
    tune_data, tune_sr = tune.raw()

    rms = librosa.feature.rms(tune_data,
                              frame_length=FitnessFunctionsConf.FRAME_LENGTH,
                              hop_length=FitnessFunctionsConf.HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(tune_data,
                                             frame_length=FitnessFunctionsConf.FRAME_LENGTH,
                                             hop_length=FitnessFunctionsConf.HOP_LENGTH)[0]

    rms_zcr = np.hstack((rms, zcr))
    original_rms_zcr = np.hstack((original_rms, original_zcr))

    return 1 / np.linalg.norm(original_rms_zcr - rms_zcr)


def signal_l2_distance(original_data: np.ndarray, original_sr: float, tune) -> float:
    tune_data, _ = tune.raw()

    return 1 / np.linalg.norm(original_data - tune_data)


def mean_mel_l2_distance(original_DB: np.ndarray, original_sr: float, tune) -> float:
    tune_data, tune_sr = tune.raw()

    tune_D = librosa.feature.melspectrogram(y=tune_data, sr=tune_sr)
    tune_DB = librosa.amplitude_to_db(tune_D, ref=np.max)
    # This is the more correct form, but as we already used the incorrect we're going with it
    # tune_DB = librosa.power_to_db(tune_D, ref=np.max)

    dist = np.linalg.norm(original_DB - tune_DB, axis=0).mean()
    if dist:
        return 1 / dist
    return np.inf
