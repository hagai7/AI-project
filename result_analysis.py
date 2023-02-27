import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import inspect
import librosa.display
from typing import List, Tuple
from time import gmtime, strftime

import configuration
from configuration import ResultAnalysisConf
from tune_generator import Tune


def create_train_fitness_plot(history: List[Tuple[Tune, float]]):
    x = np.arange(1, len(history) + 1)
    _, fitnesses = zip(*history)

    plt.figure()
    plt.plot(x, fitnesses, '-o', color='black')
    plt.title('Fitness of best in each generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')


def save_train_fitness_plot(history: List[Tuple[Tune, float]], path: str):
    create_train_fitness_plot(history)
    plt.savefig(path)


def display_train_fitness_plot(history: List[Tuple[Tune, float]]):
    create_train_fitness_plot(history)
    plt.show()


def save_configuration(path):
    cls_name_to_fields_str = {}
    conf_classes = inspect.getmembers(sys.modules[configuration.__name__], inspect.isclass)
    for cls_name, cls in conf_classes:
        field_names = [attr for attr in dir(cls)
                       if not callable(getattr(cls, attr)) and not attr.startswith("__")]
        cls_name = cls_name[:cls_name.index('Conf')]
        fields_strs = []
        for field_name in field_names:
            if callable(getattr(cls, field_name)):
                fields_strs.append(f"{field_name} = {getattr(cls, field_name).__name__}")
            else:
                fields_strs.append(f"{field_name} = {getattr(cls, field_name)}")
        cls_name_to_fields_str[cls_name] = '\n'.join(fields_strs)

    output_str = '\n'.join(f"{cls_name}:\n{cls_name_to_fields_str[cls_name]}\n"
                           for cls_name in cls_name_to_fields_str)

    with open(path, 'w') as f:
        f.write(output_str)


def save_music(tune: Tune, dir_path):
    tune.save_tune('tune', dir_path, with_timestamp=False)


def create_spectrogram(data, sr):
    D = librosa.feature.melspectrogram(y=data, sr=sr)
    DB = librosa.amplitude_to_db(D, ref=np.max)
    # This is the more correct form, but as we already used the incorrect we're going with it
    # DB = librosa.power_to_db(D, ref=np.max)

    plt.figure()
    librosa.display.specshow(DB, sr=sr, hop_length=ResultAnalysisConf.SPECTROGRAM_HOP_LENGTH, x_axis='time', y_axis='mel')


def save_spectrogram(tune: Tune, path: str):
    create_spectrogram(*tune.raw())
    plt.savefig(path)


def save_runtime(runtime_delta, dir_path):
    with open(os.path.join(dir_path, f"runtime_{str(runtime_delta)}"), 'w'):
        pass


def document_run(history: List[Tuple[Tune, float]], output_dir_path: str, input_file_name, runtime_delta=None):
    best_tune, best_fitness = history[-1]

    dir_name = f"fitness_{best_fitness}__time_{strftime('%Y-%m-%d__%H_%M_%S', gmtime())}__{input_file_name}"
    dir_path = os.path.join(output_dir_path, dir_name)
    os.mkdir(dir_path)

    save_train_fitness_plot(history, os.path.join(dir_path, 'fitness_history_plot.png'))
    save_configuration(os.path.join(dir_path, 'configuration.txt'))
    save_music(best_tune, dir_path)
    save_spectrogram(best_tune, os.path.join(dir_path, 'spectrogram.png'))
    if runtime_delta is not None:
        save_runtime(runtime_delta, dir_path)
