"""
You are welcome to mess with the configuration.
Parameters you are likely to be interested in have indicative comments.
"""

import numpy as np


class MainConf:
    POP_SIZE = 500  # Mess with me
    MUTATION_DECAY = 1  # Mess with me
    NUM_ITERATIONS = 50  # Mess with me
    MUTATION_RATE = 0.5  # Mess with me
    FITNESS_THRESHOLD = np.inf


class TuneGeneratorConf:
    MAX_AMP = 2
    MIN_AMP = 0.5
    MUTATION_FRACTION = 2

    BLOCK_REMOVE_PROB = 0.5
    NOTE_CHANGE_PROB = 0.5
    ADD_BLOCK_PROB = 0.5
    MAX_BLOCKS_TO_ADD = 1
    MIN_OFFSET = 0

    REPRODUCE_FUNC_NAME = 'reproduce_by_single_split'


class GeneticAlgorithmConf:
    TOP_FRACTION = 1 / 4  # Mess with me (it's the fraction of best individuals we keep with certainty)


class ResultAnalysisConf:
    SPECTROGRAM_HOP_LENGTH = 512


class FitnessFunctionsConf:
    N_FFT = 2048
    HOP_LENGTH = N_FFT // 4
    FRAME_LENGTH = 2048
