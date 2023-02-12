import torch
from torch.utils.data import Dataset
import pickle
__all__ = ['ReadySetGoData']


class ReadySetGoData(Dataset):
    def __init__(self, config):
        self.config = config
        if self.config.dataset.phase == 'train':
            self.path_to_data = self.config.dataset.path_to_train_data
        elif self.config.dataset.phase == 'test':
            self.path_to_data = self.config.dataset.path_to_test_data
        self.input_transform = None
        self.target_transform = None
        if config.dataset.input_transform:
            self.input_transform = eval(self.config.dataset.transform)()
        if config.dataset.target_transform:
            self.target_transform = eval(self.config.dataset.target_transform)()
        with open(self.path_to_data, 'rb') as f:
            self.all_data = pickle.load(f)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data = self.all_data[idx]
        data_x = data['data_x']
        data_y = data['data_y']
        first_cue_idx = data['first_cue_idx']
        second_cue_idx = data['second_cue_idx']
        sine_last_idx = data['sine_last_idx']

        if self.input_transform:
            data_x = self.input_transform(data_x)
        if self.target_transform:
            data_y = self.target_transform(data_y)

        return data_x, data_y, first_cue_idx, second_cue_idx, sine_last_idx


class CueTransform(object):

    def __call__(self, sample):
        sample = torch.Tensor(sample)
        sample = sample.float()
        sample = torch.transpose(sample, 0, 1)

        return sample


class ToTensor(object):

    def __call__(self, sample):
        sample = torch.tensor([sample])
        return sample.float()

