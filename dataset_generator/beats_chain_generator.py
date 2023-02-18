from data_generator import *
from other_utils import *
import argparse

parser = argparse.ArgumentParser(description='beats chain data generator')
parser.add_argument('-t', '--target-shape',
                    help='target shape could be half_sine, full_sine, double_sine,triple_sine, triangle, rectangle',
                    required=True, type=str, default='full_sine')
parser.add_argument('-n', '--number-of-data-samples', help='number of data samples', required=True, type=int)
parser.add_argument('-nb', '--number-of-beats', help='number of beats', required=True, type=int, default=10)
parser.add_argument('-nt', '--number-of-targets', help='number of targets', required=True, type=int, default=11)
parser.add_argument('-dt', '--delta-t', help='number of targets', required=True, type=float, default=0.001)
parser.add_argument('-i', '--interval-range', help='string array of intervals', required=True, type=str)
parser.add_argument('-p', '--path', help='path to save', required=True, type=str)
args = vars(parser.parse_args())
print(args)

target_shape = args['target_shape']
number_of_data_samples = args['number_of_data_samples']
number_of_beats = args['number_of_beats']
number_of_targets = args['number_of_targets']
interval_ranges = args['interval_ranges']
target_starts_from_beat = 2
data_duration = 25
first_beat_first_time = 4
first_beat_last_time = 10
target_amplitude = 1
beats_amplitude = 1
dt = args['delta_t']
path_to_save = args['path']

if __name__ == "__main__":
    all_data = []
    for i in range(number_of_data_samples):
        data_x, data_y, first_beat_idx, interval_between_beats, data_y_start_idx, data_y_end_idx = \
            beats_chain_generator(
                number_of_beats=number_of_beats,
                number_of_targets=number_of_targets,
                target_starts_from_beat=target_starts_from_beat,
                target_shape=target_shape,
                data_duration=data_duration,
                first_beat_first_time=first_beat_first_time,
                first_beat_last_time=first_beat_last_time,
                interval_ranges=interval_ranges,
                target_amplitude=target_amplitude,
                beats_amplitude=beats_amplitude,
                dt=dt)

        data = {'data_x': data_x,
                'data_y': data_y,
                'first_beat_idx': first_beat_idx,
                'interval_between_beats': interval_between_beats,
                'data_y_start_idx': data_y_start_idx,
                'data_y_end_idx': data_y_end_idx,
                'dt': dt}

        all_data.append(data)

    data_name = f'beats_chain_generator_{number_of_data_samples}_duration{data_duration}_dt{dt}_' \
                f'beat_f{first_beat_first_time}l{first_beat_last_time}_intervalranges_{interval_ranges}_' \
                f'targeta{target_amplitude}_beatsa{beats_amplitude}_targetshape{target_shape}_beatsn{number_of_beats}' \
                f'_targetsn{number_of_targets}'
    save(path_to_save, data_name, all_data)
