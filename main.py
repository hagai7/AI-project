import sys
import os.path
import librosa
import numpy as np
from functools import partial
from pathlib import Path
from time import time

from tune_generator import Tune, TuneData, generate_population
from genetic_algorithm import GeneticAlgorithm
from result_analysis import document_run, save_music
from configuration import MainConf, FitnessFunctionsConf
from fitness_functions import mean_mel_l2_distance, zcr_rms_distance
from utils import log_info

USAGE_MSG = f"Usage: python3 {os.path.basename(__file__)} input_wav_path output_directory " \
            f"[pop_size iterations]"
ANALYZE_RESULTS = False


def validate_args(input_path, output_dir_path):
    if not os.path.exists(input_path) or not os.path.isfile(input_path):
        return False, 'Invalid input path'

    if not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
        return False, 'Invalid output dir'

    return True, ''


def main():
    # Input validation
    if len(sys.argv) == 3:
        input_path, output_dir_path = sys.argv[1:3]
    elif len(sys.argv) == 5:
        input_path, output_dir_path, pop_size_str, iterations_str = sys.argv[1:5]
        MainConf.POP_SIZE = int(pop_size_str)
        MainConf.NUM_ITERATIONS = int(iterations_str)
    else:
        print(USAGE_MSG)
        return

    are_args_valid, error_msg = validate_args(input_path, output_dir_path)
    if not are_args_valid:
        print(error_msg)
        return

    log_info(f"Population size {MainConf.POP_SIZE}, iterations {MainConf.NUM_ITERATIONS}")

    # Load input audio
    orig_tune, orig_sr = librosa.load(input_path)

    # Initialize values and building blocks database
    Tune.set_sr(orig_sr)
    Tune.set_tune_len(len(orig_tune))
    TuneData.init_db()

    # Compute original mel-spectrogram for fitness function
    orig_D = librosa.feature.melspectrogram(y=orig_tune, sr=orig_sr)
    orig_DB = librosa.amplitude_to_db(orig_D, ref=np.max)
    # This is the more correct form, but as we already used the incorrect we're going with it
    # orig_DB = librosa.power_to_db(orig_D, ref=np.max)

    # Compute original zcr and mrs for fitness function
    orig_rms = librosa.feature.rms(orig_tune,
                                   frame_length=FitnessFunctionsConf.FRAME_LENGTH,
                                   hop_length=FitnessFunctionsConf.HOP_LENGTH)[0]
    orig_zcr = librosa.feature.zero_crossing_rate(orig_tune,
                                                  frame_length=FitnessFunctionsConf.FRAME_LENGTH,
                                                  hop_length=FitnessFunctionsConf.HOP_LENGTH)[0]

    mel_fitness_func = partial(mean_mel_l2_distance, orig_DB, orig_sr)
    zcr_rms_fitness_func = partial(zcr_rms_distance, orig_rms, orig_zcr)

    ga = GeneticAlgorithm(fitness_func=mel_fitness_func, population_generation_func=generate_population,
                          fitness_threshold=MainConf.FITNESS_THRESHOLD, mutation_rate=MainConf.MUTATION_RATE,
                          num_iterations=MainConf.NUM_ITERATIONS, mutation_decay=MainConf.MUTATION_DECAY)

    start_time = time()
    best_history = list(ga.generate(pop_size=MainConf.POP_SIZE))
    end_time = time()

    if ANALYZE_RESULTS:
        document_run(best_history, output_dir_path, Path(input_path).stem, end_time - start_time)
    else:
        save_music(best_history[-1][0], output_dir_path)


if __name__ == '__main__':
    main()
