import torch
import torch.nn as nn

class HighwayMLP(nn.Module):

    def __init__(
        self,
        input_size,
        # gate_bias=-2,
        gate_activation=nn.Sigmoid()
        ):
        super(HighwayMLP, self).__init__()
        self.gate_activation = gate_activation
        self.gate_layer = nn.Linear(input_size*2, input_size)
        # self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, original, additional):

        concat = torch.cat([original, additional], 2)
        gate = self.gate_activation(self.gate_layer(concat))

        # original-aware gating on additinal information
        additional = torch.mul(gate, additional)
        original = torch.mul((1 - gate), original)

        return torch.add(original, additional)
