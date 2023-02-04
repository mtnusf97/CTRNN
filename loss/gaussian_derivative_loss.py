import torch

__all__ = ['gaussian_derivative_loss']


def gaussian_derivative_loss(model_outputs,
                                         targets):
    """
    :param model_outputs: List of tensors of outputs of the model in each time step, shape is
    [batch_size, output_size, time_steps]
    :param targets: List of tensors of targets in each time step, shape is similar to model_outputs
    :return:
    """
    loss = targets * sigmoid(model_outputs) + (1 - targets) * (1 - sigmoid(model_outputs))
    loss = loss[torch.abs(loss - 0.50) > 0.001]
    loss = torch.sum(loss) / loss.size()[0]
    return loss


def sigmoid(x):
    return 1/(1 + torch.exp(-x))

