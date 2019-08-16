import torch
import torch.nn as nn

class HighwayMLP(nn.Module):

    def __init__(
        self,
        input_size,
        gate_bias=-2,
        input_activation=nn.ReLU(),
        gate_activation=nn.Sigmoid()
        ):

        super(HighwayMLP, self).__init__()

        self.input_activation = input_activation
        self.gate_activation = gate_activation

        self.input_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x, y):

        input_proj = self.input_activation(self.input_layer(x))
        gate = self.gate_activation(self.gate_layer(y))

        input_proj_gated = torch.mul(input_proj, gate)
        input_gated = torch.mul((1 - gate), x)

        return torch.add(input_gated, input_proj_gated)