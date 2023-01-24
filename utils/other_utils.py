import pickle
import os


def save(path_to_save, file_name, object_to_save):
    with open(os.path.join(path_to_save, file_name) + '.pkl', 'wb') as file:
        pickle.dump(object_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)
    return 1


def load(path_to_file):
    with open(path_to_file, 'rb') as file:
        rnn_params = pickle.load(file)
    return rnn_params
