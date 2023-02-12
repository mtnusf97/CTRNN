import torch.nn as nn

__all__ = ['beats_chain_mse_loss']


def beats_chain_mse_loss(model_outputs,
                         targets,
                         all_first_beat_idx,
                         all_first_target_idx,
                         all_last_target_idx,
                         config):
    """
    :param model_outputs: List of tensors of outputs of the model in each time step, shape is
    [batch_size, output_size, time_steps]
    :param targets: List of tensors of targets in each time step, shape is similar to model_outputs
    :param all_first_beat_idx: Tensor of first beat indices for each training data, shape is [batch_size]
    :param all_first_target_idx: Tensor of second beat indices for each training data, shape is [batch_size]
    :param all_last_target_idx: Tensor of sine last indices for each training data, shape is [batch_size]
    :param config: This training config file
    :return:
    """
    criterion = nn.MSELoss()
    if config.train.loss_includes == 'all':
        return criterion(model_outputs, targets)

    if config.train.loss_includes == 'just_target':
        batches_loss = 0
        for i in range(model_outputs.shape[0]):
            batches_loss += criterion(model_outputs[i, :, all_first_target_idx[i]:all_last_target_idx[i]],
                                      targets[i, :, all_first_target_idx[i]:all_last_target_idx[i]])
        return batches_loss / model_outputs.shape[0]

    if config.train.loss_includes == 'target_and_interval':
        batches_loss = 0
        for i in range(model_outputs.shape[0]):
            batches_loss += criterion(model_outputs[i, :, all_first_beat_idx[i]:all_last_target_idx[i]],
                                      targets[i, :, all_first_beat_idx[i]:all_last_target_idx[i]])
        return batches_loss / model_outputs.shape[0]
