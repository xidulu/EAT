import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute_relative_changes(a):
    """
    Compute the relative changes of a sequence with a moving average window.
    """
    return np.abs(
        (a[1:] - a[:-1])
    )


def get_early_stop_point_posthoc(trace, threshold, min_distance=25, normalize=False):
    trace = trace[1:]
    for i, value in enumerate(trace):
        if i < min_distance:
            continue
        if value < threshold:
            return i
    return None

def exponential_moving_average(data, timescale):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = timescale * data[i] + (1 - timescale) * ema[i - 1]
    
    return ema

def exponential_moving_variance(data, timescale, ema_0=0.0, normalize=False):
    ema_mean = exponential_moving_average(data, timescale)
    ema_variance = np.zeros_like(data)
    ema_variance[0] = ema_0
    
    for i in range(1, len(data)):
        deviation = data[i] - ema_mean[i]
        ema_variance[i] = timescale * (deviation ** 2) + (1 - timescale) * ema_variance[i - 1]

    return ema_variance

def simple_moving_average_variance(data, window_size=10):
    window = np.ones(window_size) / window_size
    mean_x = np.convolve(data, window, mode='valid')
    mean_x_sq = np.convolve(data**2, window, mode='valid')
    moving_var = mean_x_sq - mean_x**2
    # Pad the beginning of the moving variance to match the length of the original data
    moving_var = np.concatenate((np.zeros(window_size - 1), moving_var))
    return moving_var

def choose_early_stop_point(entropy, timescale, threshold, min_distance=25, ema_0=0.0, min_value_threshold=None, normalize=False):
    ema_variance = exponential_moving_variance(entropy, timescale, ema_0, normalize=normalize)
    if min_distance > len(entropy):
        return -1
    if min_value_threshold is not None:
        min_value_pos = np.argmax(entropy < min_value_threshold)
        min_distance = max(min_value_pos, min_distance)
    exit_idx = get_early_stop_point_posthoc(ema_variance, threshold, min_distance, normalize=normalize)
    return exit_idx if exit_idx is not None else -1

def choose_early_stop_point_moving_window(entropy, window_size, threshold):
    if window_size > len(entropy):
        return -1
    ema_variance = simple_moving_average_variance(entropy, window_size)
    # ema_variance = ema_variance / exponential_moving_average(entropy, timescale)
    exit_idx = get_early_stop_point_posthoc(ema_variance, threshold, window_size, normalize=False)
    return exit_idx if exit_idx is not None else -1