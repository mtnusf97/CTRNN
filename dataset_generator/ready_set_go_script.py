from data_generator import *
from other_utils import *
import argparse

parser = argparse.ArgumentParser(description='ready set go data generator')
parser.add_argument('-t', '--target-shape', help='target shape could be half_sine, full_sine, double_sine, '
                                                 'triple_sine, triangle, rectangle', required=True, type=str)
parser.add_argument('-n', '--number-of-data-samples', help='number of data samples', required=True, type=int)
parser.add_argument('-p', '--path', help='path to save', required=True, type=str)
args = vars(parser.parse_args())
print(args)

target_shape = args['target_shape']
number_of_data_samples = args['number_of_data_samples']
data_duration = 40
first_cue_first_time = 10
first_cue_last_time = 20
min_interval = 2
max_interval = 8
target_amplitude = 1
cue_amplitude = 1
dt = 0.01
path_to_save = args['path']

if __name__ == "__main__":
    all_data = []
    for i in range(number_of_data_samples):
        data_x, data_y, first_cue_idx, second_cue_idx, sine_last_idx = ready_set_go_generator(target_shape,
                                                                                              data_duration,
                                                                                              first_cue_first_time,
                                                                                              first_cue_last_time,
                                                                                              min_interval,
                                                                                              max_interval,
                                                                                              target_amplitude,
                                                                                              cue_amplitude,
                                                                                              dt)
        data = {'data_x': data_x,
                'data_y': data_y,
                'first_cue_idx': first_cue_idx,
                'second_cue_idx': second_cue_idx,
                'sine_last_idx': sine_last_idx,
                'dt': dt}

        all_data.append(data)
    data_name = f'ready_set_go_samples_{number_of_data_samples}_duration{data_duration}_dt{dt}_' \
                f'cue_f{first_cue_first_time}l{first_cue_last_time}_interval_{min_interval}_{max_interval}_' \
                f'targeta{target_amplitude}_cuea{cue_amplitude}_targetshape{target_shape}'
    save(path_to_save, data_name, all_data)
