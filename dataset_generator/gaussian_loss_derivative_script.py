# import sys
# path_dataset = '/home/matin/McMaster/trimba/phd_codes/CTRNN/dataset'
# path_utils = '/home/matin/McMaster/trimba/phd_codes/CTRNN/utils'
# sys.path.append(path_dataset)
# sys.path.append(path_utils)

from data_generator import *
from CTRNN.utils.other_utils import *


number_of_data_samples = 99
data_duration = 40
first_cue_first_time = 10
first_cue_last_time = 20
min_interval = 2
max_interval = 8
variance = 0.4
scale = 0.99
cue_amplitude = 1
dt = 0.01

if __name__ == "__main__":
    all_data = []
    for i in range(number_of_data_samples):
        data_x, data_y, first_cue_idx, second_cue_idx, concavity = gaussian_derivative_loss_data(data_duration,
                                                                                                 first_cue_first_time,
                                                                                                 first_cue_last_time,
                                                                                                 min_interval,
                                                                                                 max_interval,
                                                                                                 cue_amplitude,
                                                                                                 variance,
                                                                                                 scale,
                                                                                                 dt)

        data = {'data_x': data_x,
                'data_y': data_y,
                'first_cue_idx': first_cue_idx,
                'second_cue_idx': second_cue_idx,
                'concavity_change': concavity,
                'variance': variance,
                'scale': scale,
                'dt': dt}

        all_data.append(data)

    path_to_save = '/home/matin/McMaster/trimba/phd_codes/CTRNN/generated_datasets'
    data_name = f'gaussian_loss_shape_samples_{number_of_data_samples}_duration_40_dt01_cue_f10_l20_interval_2_8_sa1_cuea1_variance0.4_scale0.99'
    save(path_to_save, data_name, all_data)