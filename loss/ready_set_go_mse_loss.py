import torch.nn as nn

__all__ = ['ready_set_go_mse_loss']


def ready_set_go_mse_loss(model_outputs,
                          targets,
                          all_first_cue_idx,
                          all_second_cue_idx,
                          all_sine_last_idx,
                          config):
    """
    :param model_outputs: List of tensors of outputs of the model in each time step, shape is
    [batch_size, output_size, time_steps]
    :param targets: List of tensors of targets in each time step, shape is similar to model_outputs
    :param all_first_cue_idx: Tensor of first cue indices for each training data, shape is [batch_size]
    :param all_second_cue_idx: Tensor of second cue indices for each training data, shape is [batch_size]
    :param all_sine_last_idx: Tensor of sine last indices for each training data, shape is [batch_size]
    :param config: This training config file
    :return:
    """
    criterion = nn.MSELoss()
    if config.train.loss_includes == 'all':
        return criterion(model_outputs, targets)

    if config.train.loss_includes == 'just_sine':
        batches_loss = 0
        for i in range(model_outputs.shape[0]):
            batches_loss += criterion(model_outputs[i, :, all_second_cue_idx[i]:all_sine_last_idx[i]],
                                      targets[i, :, all_second_cue_idx[i]:all_sine_last_idx[i]])
        return batches_loss / model_outputs.shape[0]

    if config.train.loss_includes == 'sine_and_interval':
        batches_loss = 0
        for i in range(model_outputs.shape[0]):
            batches_loss += criterion(model_outputs[i, :, all_first_cue_idx[i]:all_sine_last_idx[i]],
                                      targets[i, :, all_first_cue_idx[i]:all_sine_last_idx[i]])
        return batches_loss / model_outputs.shape[0]