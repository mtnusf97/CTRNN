from torch.utils.data import Dataset
import pickle


class GaussianDerivativeLossData(Dataset):
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
        concavity_change = data['concavity_change']

        if self.input_transform:
            data_x = self.input_transform(data_x)
        if self.target_transform:
            data_y = self.target_transform(data_y)

        return data_x, data_y, first_cue_idx, second_cue_idx, concavity_change
