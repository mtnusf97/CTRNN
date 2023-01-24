import torch.nn as nn

__all__ = ['cue_mse_loss']


def cue_mse_loss(model_outputs, targets, all_cue_idx, all_sine_last_idx, config):
    """
    :param model_outputs: List of tensors of outputs of the model in each time step, shape is
    [batch_size, output_size, time_steps]
    :param targets: List of tensors of targets in each time step, shape is similar to model_outputs
    :param all_cue_idx: Tensor of cue indices for each training data, shape is [batch_size]
    :param all_sine_last_idx: Tensor of sie last indices for each training data, shape is [batch_size]
    :param config: This training config file
    :return:
    """
    criterion = nn.MSELoss()
    if config.train.pre_is_in_loss:
        if config.train.post_is_in_loss:
            return criterion(model_outputs, targets)
        else:
            batches_loss = 0
            for i in range(model_outputs.shape[0]):
                batches_loss += criterion(model_outputs[i, :, :all_sine_last_idx[i]],
                                          targets[i, :, :all_sine_last_idx[i]])
            return batches_loss / model_outputs.shape[0]
    elif not config.train.pre_is_in_loss:
        if config.train.post_is_in_loss:
            batches_loss = 0
            for i in range(model_outputs.shape[0]):
                batches_loss += criterion(model_outputs[i, :, all_cue_idx[i]:],
                                          targets[i, :, all_cue_idx[i]:])
            return batches_loss / model_outputs.shape[0]
        elif not config.train.post_is_in_loss:
            batches_loss = 0
            for i in range(model_outputs.shape[0]):
                batches_loss += criterion(model_outputs[i, :, all_cue_idx[i]:all_sine_last_idx[i]],
                                          targets[i, :, all_cue_idx[i]:all_sine_last_idx[i]])
            return batches_loss / model_outputs.shape[0]
