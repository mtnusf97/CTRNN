import torch
import torch.nn as nn

__all__ = ['PCTRNN']


class PCTRNN(nn.Module):

    def __init__(self, config):
        super(PCTRNN, self).__init__()

        self.input_size = config.model.input_size
        self.input_perceptron_size = config.model.input_perceptron_size
        self.ctrnn_size = config.model.ctrnn_size
        self.output_perceptron_size = config.model.output_perceptron_size
        self.output_size = config.model.output_size
        self.has_feedback = config.model.has_feedback
        self.dt = config.model.dt  # time between Euler integration updates (seconds)
        self.tau = config.model.tau  # "neuronal" time constant
        self.dt_over_tau = self.dt / self.tau
        self.delay = config.model.delay  # the delay of the output effect on input (seconds)
        self.delay_index = int(self.delay / self.dt) + 1

        self.input_perceptron = nn.Linear(in_features=self.input_size, out_features=self.input_perceptron_size,
                                          bias=True)
        self.wI = nn.Linear(in_features=self.input_perceptron_size, out_features=self.ctrnn_size, bias=True)  # input matrix
        self.wR = nn.Linear(in_features=self.ctrnn_size, out_features=self.ctrnn_size, bias=False)  # recurrent matrix
        self.output_perceptron = nn.Linear(in_features=self.ctrnn_size, out_features=self.output_perceptron_size,
                                           bias=True)
        self.wO = nn.Linear(in_features=self.output_perceptron_size, out_features=self.output_size, bias=True)  # output matrix
        if self.has_feedback:  # feedback matrix
            self.wF = nn.Linear(in_features=self.output_size, out_features=self.ctrnn_size, bias=False)

        self.device = config.device

    def forward(self, X, activations, hidden_states, outputs):
        """
        :param X: Tensor of inputs to the network, shape is [batch_size, input_size, time_steps]
        :param activations: Tensor of initial activations of the network, shape is [batch_size, ctrnn_size]
        :param hidden_states: tanh of init_activations
        :param outputs: List of Tensors of initial outputs of the network, shape is [delay_size + 1, batch_size,
        output_size]
        :return: Networks final outputs, activations, and hidden_states
        """
        batch_size = X.shape[0]
        time_steps = X.shape[-1]
        for i in range(time_steps):
            x = X[:, :, i]
            x = self.input_perceptron(x)
            x = torch.tanh(x)
            if self.has_feedback:
                delayed_output = outputs[-self.delay_index]
                da_over_dt = -activations + self.wI(x) + self.wR(hidden_states) + self.wF(delayed_output)
            else:
                da_over_dt = -activations + self.wI(x) + self.wR(hidden_states)

            activations = activations + self.dt_over_tau * da_over_dt
            hidden_states = torch.tanh(activations)
            last_output = self.output_perceptron(hidden_states)
            last_output = torch.tanh(last_output)
            last_output = self.wO(last_output)
            outputs.append(last_output)

        tensor_outputs = torch.cat([i.view(batch_size, self.output_size, 1) for i in outputs[self.delay_index:]],
                                   dim=2)

        return tensor_outputs, activations, hidden_states

    def init_activations_outputs(self, batch_size, for_predict=False):
        activations = torch.zeros((batch_size, self.ctrnn_size)).to(self.device)
        if for_predict:
            activations = torch.zeros(self.ctrnn_size)
        hidden_states = torch.tanh(activations).to(self.device)
        outputs = [torch.zeros((batch_size, self.output_size))] * (self.delay_index - 1)
        if for_predict:
            outputs = [torch.zeros(self.output_size)] * (self.delay_index - 1)
        outputs = [output.to(self.device) for output in outputs]
        last_output = self.output_perceptron(hidden_states)
        last_output = torch.tanh(last_output)
        last_output = self.wO(last_output)
        outputs.append(last_output)

        return activations, hidden_states, outputs

    def predict(self, x, activations, hidden_states, outputs):
        """
        :param x: Tensor of input to the network, shape is [input_size, time_steps]
        :param activations:
        :param hidden_states:
        :param outputs:
        :return:
        """
        all_activations = []
        all_hidden_states = []
        time_steps = x.shape[-1]
        for i in range(time_steps):
            x_i = x[:, i]
            x_i = self.input_perceptron(x_i)
            x_i = torch.tanh(x_i)
            if self.has_feedback:
                delayed_output = outputs[-self.delay_index]
                da_over_dt = -activations + self.wI(x_i) + self.wR(hidden_states) + self.wF(delayed_output)
            else:
                da_over_dt = -activations + self.wI(x_i) + self.wR(hidden_states)

            activations = activations + self.dt_over_tau * da_over_dt
            hidden_states = torch.tanh(activations)
            last_output = self.output_perceptron(hidden_states)
            last_output = torch.tanh(last_output)
            last_output = self.wO(last_output)
            all_activations.append(activations)
            all_hidden_states.append(hidden_states)
            outputs.append(last_output)

        tensor_outputs = torch.cat([i for i in outputs[self.delay_index:]], dim=0)

        return tensor_outputs, all_activations, all_hidden_states

    def old_predict(self, x, activations, hidden_states, outputs):
        """
        :param x: Tensor of input to the network, shape is [input_size, time_steps]
        :param activations:
        :param hidden_states:
        :param outputs:
        :return:
        """
        time_steps = x.shape[-1]
        for i in range(time_steps):
            x_i = x[:, i]
            x_i = self.input_perceptron(x_i)
            x_i = torch.tanh(x_i)
            if self.has_feedback:
                delayed_output = outputs[-self.delay_index]
                da_over_dt = -activations + self.wI(x_i) + self.wR(hidden_states) + self.wF(delayed_output)
            else:
                da_over_dt = -activations + self.wI(x_i) + self.wR(hidden_states)

            activations = activations + self.dt_over_tau * da_over_dt
            hidden_states = torch.tanh(activations)
            last_output = self.output_perceptron(hidden_states)
            last_output = torch.tanh(last_output)
            last_output = self.wO(last_output)
            outputs.append(last_output)

        tensor_outputs = torch.cat([i for i in outputs[self.delay_index:]], dim=0)

        return tensor_outputs, activations, hidden_states
