from __future__ import print_function, division
import numpy as np
import torch
import scipy


# from scipy import signal


def sin_generator(num_of_points, data_points_interval):
    data_x = np.arange(num_of_points) * data_points_interval
    data_x = data_x.reshape(1, -1).reshape(-1, 1)
    data_y = np.sin(data_x)
    return data_x, data_y


def simple_steady_beat_generator(input_duration, output_duration, data_points_interval, beats_interval):
    """Generate simple steady beats

    Arguments:
      input_duration: duration of the networks input (seconds)
      output_duration: duration of the networks output (seconds)
      data_points_interval: time difference between every two consecutive data points ==> dt (seconds)
      beats_interval: time difference between every two consecutive beats (seconds)

    Returns:
      data_x: input to the network
      data_y: output to the network

    """

    output_size = int(output_duration / data_points_interval)
    input_size = int(input_duration / data_points_interval)
    beats_interval_size = int(beats_interval / data_points_interval)
    data_x = np.zeros(output_size)
    data_x[input_size:output_size] = -1
    idx = 0
    while idx <= input_size:
        data_x[idx] = 1
        idx += beats_interval_size

    data_y = np.zeros(output_size)
    idx = 0
    while idx < output_size:
        data_y[idx] = 1
        idx += beats_interval_size

    data_x = data_x.reshape(1, -1).reshape(-1, 1)
    data_y = data_y.reshape(1, -1).reshape(-1, 1)
    return data_x, data_y


def cue_signal_generator(data_duration,
                         cue_first_time,
                         cue_last_time,
                         sine_length,
                         sine_amplitude,
                         dt):
    assert cue_last_time + sine_length <= data_duration
    data_size = int(data_duration / dt)

    # create data_x
    data_x = torch.zeros((1, data_size))
    cue_time = np.random.uniform(low=cue_first_time, high=cue_last_time)
    cue_idx = int(cue_time / dt)
    data_x[0, cue_idx] = 1.0

    # create data_y
    data_y = torch.zeros((1, data_size))
    sine_size = int(sine_length / dt)
    sine_times = torch.arange(sine_size) * (2 * np.pi / sine_size)
    sine_wave = torch.sin(sine_times) * sine_amplitude
    data_y[0, cue_idx:cue_idx + sine_size] = sine_wave
    sine_last_idx = cue_idx + sine_size

    return data_x, data_y, cue_idx, sine_last_idx


def ready_set_go_generator(target_shape,
                           data_duration,
                           first_cue_first_time,
                           first_cue_last_time,
                           min_interval,
                           max_interval,
                           target_amplitude,
                           cue_amplitude,
                           dt):
    """
    :param target_shape: str showing the shape of target, could be one of options: triangle, rectangle, semicircle
    , half_sine, full_sine, double_sine, triple_sine
    :param data_duration:
    :param first_cue_first_time:
    :param first_cue_last_time:
    :param min_interval:
    :param max_interval:
    :param target_amplitude:
    :param cue_amplitude:
    :param dt:
    :return:
    """
    assert first_cue_last_time + 2 * max_interval <= data_duration
    assert min_interval < max_interval

    data_size = int(data_duration / dt)

    # create data_x
    data_x = torch.zeros((1, data_size))
    first_cue_time = np.random.uniform(low=first_cue_first_time, high=first_cue_last_time)
    interval_between_cues = np.random.uniform(low=min_interval, high=max_interval)
    first_cue_idx = int(first_cue_time / dt)
    second_cue_idx = int((first_cue_time + interval_between_cues) / dt)
    data_x[0, first_cue_idx] = data_x[0, second_cue_idx] = cue_amplitude

    # create data_y
    data_y = torch.zeros((1, data_size))
    target_size = int(interval_between_cues / dt)
    if target_shape == 'half_sine':
        sine_times = torch.arange(target_size) * (np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = sine_wave
    elif target_shape == 'full_sine':
        sine_times = torch.arange(target_size) * (2 * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = sine_wave
    elif target_shape == 'double_sine':
        sine_times = torch.arange(target_size) * (4 * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = sine_wave
    elif target_shape == 'triple_sine':
        sine_times = torch.arange(target_size) * (6 * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = sine_wave
    elif target_shape == 'triangle':
        triangle_times = torch.arange(target_size) * (2 * np.pi / target_size)
        triangle_wave = 0.5 * (scipy.signal.sawtooth(triangle_times, 0.5) + 1) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = torch.tensor(triangle_wave)
    elif target_shape == 'rectangle':
        data_y = torch.zeros((1, data_size))
        rectangle_size = int(interval_between_cues / dt)
        data_y[0, second_cue_idx:second_cue_idx + rectangle_size] = target_amplitude
    else:
        raise Exception(f"target shape {target_shape} is not defined")

    target_last_idx = second_cue_idx + target_size

    return data_x, data_y, first_cue_idx, second_cue_idx, target_last_idx


def ready_set_go_forced_generator(target_shape,
                                  data_duration,
                                  first_cue_time,
                                  second_cue_time,
                                  target_amplitude,
                                  cue_amplitude,
                                  dt):
    """
    :param target_shape: str showing the shape of target, could be one of options: triangle, rectangle, semicircle
    , half_sine, full_sine, double_sine, triple_sine
    :param data_duration:
    :param first_cue_time:
    :param second_cue_time:
    :param target_amplitude:
    :param cue_amplitude:
    :param dt:
    :return:

    """
    assert 2 * second_cue_time - first_cue_time < data_duration

    data_size = int(data_duration / dt)

    # create data_x
    data_x = torch.zeros((1, data_size))
    first_cue_idx = int(first_cue_time / dt)
    second_cue_idx = int(second_cue_time / dt)
    data_x[0, first_cue_idx] = data_x[0, second_cue_idx] = cue_amplitude

    # create data_y
    interval_between_cues = second_cue_time - first_cue_time
    data_y = torch.zeros((1, data_size))
    target_size = int(interval_between_cues / dt)
    if target_shape == 'half_sine':
        sine_times = torch.arange(target_size) * (np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = sine_wave
    elif target_shape == 'full_sine':
        sine_times = torch.arange(target_size) * (2 * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = sine_wave
    elif target_shape == 'double_sine':
        sine_times = torch.arange(target_size) * (4 * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = sine_wave
    elif target_shape == 'triple_sine':
        sine_times = torch.arange(target_size) * (6 * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = sine_wave
    elif target_shape == 'triangle':
        triangle_times = torch.arange(target_size) * (2 * np.pi / target_size)
        triangle_wave = 0.5 * (scipy.signal.sawtooth(triangle_times, 0.5) + 1) * target_amplitude
        data_y[0, second_cue_idx:second_cue_idx + target_size] = torch.tensor(triangle_wave)
    elif target_shape == 'rectangle':
        data_y = torch.zeros((1, data_size))
        rectangle_size = int(interval_between_cues / dt)
        data_y[0, second_cue_idx:second_cue_idx + rectangle_size] = target_amplitude
    else:
        raise Exception(f"target shape {target_shape} is not defined")

    target_last_idx = second_cue_idx + target_size

    return data_x, data_y, first_cue_idx, second_cue_idx, target_last_idx


def gaussian_derivative_shape(x, concavity_change_point, variance):
    b = concavity_change_point
    c = variance
    return np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2))) * -2 * (x - b) / (2 * np.power(c, 2))


def scaled_gaussian_derivative_shape(x, concavity_change_point, variance, scale):
    b = concavity_change_point
    c = variance
    return (scale / (2 * gaussian_derivative_shape(b - c, b, c))) * gaussian_derivative_shape(x, b, c) + 0.5


def gaussian_derivative_loss_generator(data_duration,
                                       first_cue_first_time,
                                       first_cue_last_time,
                                       min_interval,
                                       max_interval,
                                       cue_amplitude,
                                       variance,
                                       scale,
                                       dt):
    assert first_cue_last_time + 2 * max_interval <= data_duration
    assert min_interval < max_interval

    data_size = int(data_duration / dt)

    # create data_x
    data_x = torch.zeros((1, data_size))
    first_cue_time = np.random.uniform(low=first_cue_first_time, high=first_cue_last_time)
    interval_between_cues = np.random.uniform(low=min_interval, high=max_interval)
    first_cue_idx = int(first_cue_time / dt)
    second_cue_idx = int((first_cue_time + interval_between_cues) / dt)
    data_x[0, first_cue_idx] = data_x[0, second_cue_idx] = cue_amplitude

    # create data_y
    concavity_change = first_cue_time + 2 * interval_between_cues
    temp_x = torch.arange(data_size).reshape((1, data_size)) * dt
    data_y = scaled_gaussian_derivative_shape(temp_x, concavity_change, variance, scale)

    return data_x, data_y, first_cue_idx, second_cue_idx, concavity_change


def gaussian_derivative_loss_forced_generator(data_duration,
                                              first_cue_time,
                                              second_cue_time,
                                              cue_amplitude,
                                              variance,
                                              scale,
                                              dt):
    assert 2 * second_cue_time - first_cue_time < data_duration

    data_size = int(data_duration / dt)

    # create data_x
    data_x = torch.zeros((1, data_size))
    first_cue_idx = int(first_cue_time / dt)
    second_cue_idx = int(second_cue_time / dt)
    data_x[0, first_cue_idx] = data_x[0, second_cue_idx] = cue_amplitude

    # create data_y
    interval_between_cues = second_cue_time - first_cue_time
    concavity_change = first_cue_time + 2 * interval_between_cues
    temp_x = torch.arange(data_size).reshape((1, data_size)) * dt
    data_y = scaled_gaussian_derivative_shape(temp_x, concavity_change, variance, scale)

    return data_x, data_y, first_cue_idx, second_cue_idx, concavity_change


def beats_chain_generator(number_of_beats=10,
                          number_of_targets=10,
                          target_starts_from_beat=2,
                          target_shape='full_sine',
                          data_duration=50,
                          first_beat_first_time=10,
                          first_beat_last_time=30,
                          min_interval=0.4,
                          max_interval=1.2,
                          target_amplitude=1,
                          beats_amplitude=1,
                          dt=0.01):
    """
    Args:
        number_of_beats:
        number_of_targets:
        target_starts_from_beat:
        target_shape:
        data_duration:
        first_beat_first_time:
        first_beat_last_time:
        min_interval:
        max_interval:
        target_amplitude:
        beats_amplitude:
        dt:

    Returns:

    """
    assert first_beat_last_time + number_of_beats * max_interval < data_duration
    assert min_interval < max_interval

    data_size = int(data_duration / dt)

    # create data_x
    data_x = torch.zeros((1, data_size))
    first_beat_time = np.random.uniform(low=first_beat_first_time, high=first_beat_last_time)
    interval_between_beats = np.random.uniform(low=min_interval, high=max_interval)
    first_beat_idx = int(first_beat_time / dt)
    for i in range(number_of_beats):
        data_x[0, int((first_beat_time + i * interval_between_beats) / dt)] = beats_amplitude

    # create data_y
    data_y = torch.zeros((1, data_size))
    target_size = int((interval_between_beats * (number_of_targets - 1)) / dt)
    data_y_start_time = first_beat_time + (target_starts_from_beat - 1) * interval_between_beats
    data_y_start_idx = int(data_y_start_time / dt)
    data_y_end_time = data_y_start_time + (number_of_targets - 1) * interval_between_beats
    data_y_end_idx = int(data_y_end_time / dt)
    if target_shape == 'half_sine':
        sine_times = torch.arange(target_size) * ((number_of_targets - 1) * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, data_y_start_idx:data_y_start_idx + len(sine_times)] = sine_wave
    elif target_shape == 'full_sine':
        sine_times = torch.arange(target_size) * (2 * (number_of_targets - 1) * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, data_y_start_idx:data_y_start_idx + len(sine_times)] = sine_wave
    elif target_shape == 'double_sine':
        sine_times = torch.arange(target_size) * (4 * (number_of_targets - 1) * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, data_y_start_idx:data_y_start_idx + len(sine_times)] = sine_wave
    elif target_shape == 'triple_sine':
        sine_times = torch.arange(target_size) * (6 * (number_of_targets - 1) * np.pi / target_size)
        sine_wave = torch.sin(sine_times) * target_amplitude
        data_y[0, data_y_start_idx:data_y_start_idx + len(sine_times)] = sine_wave
    elif target_shape == 'triangle':
        triangle_times = torch.arange(target_size) * (2 * (number_of_targets - 1) * np.pi / target_size)
        triangle_wave = 0.5 * (scipy.signal.sawtooth(triangle_times, 0.5) + 1) * target_amplitude
        data_y[0, data_y_start_idx:data_y_start_idx + len(triangle_times)] = torch.tensor(triangle_wave)
    else:
        raise Exception(f"target shape {target_shape} is not defined")

    return data_x, data_y, first_beat_idx, interval_between_beats, data_y_start_idx, data_y_end_idx
