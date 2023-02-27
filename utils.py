import numpy as np

# Logging modes
INFO = 0
DEBUG = 1

# This is the configuration - what kind of logs to enable/disable
LOG_CONFIGURATION = {INFO: True, DEBUG: False}

LOG_MODE_STR = {INFO: 'INFO', DEBUG: 'DEBUG'}


def log(mode, msg):
    if LOG_CONFIGURATION[mode]:
        print(f"{LOG_MODE_STR[mode]}: {msg}")


def log_info(msg):
    log(INFO, msg)


def log_debug(msg):
    log(DEBUG, msg)


def weighted_choice(choose_from, weights: np.ndarray, size: int):
    exp_weights = np.exp(weights)
    return np.random.choice(choose_from, size=size, replace=False, p=exp_weights/exp_weights.sum())

