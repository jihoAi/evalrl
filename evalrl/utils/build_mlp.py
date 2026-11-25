import math
import torch.nn as nn

def build_mlp(input_dim: int, output_dim: int, hidden: tuple[int, ...],
              activation=nn.ReLU, out_activation=nn.Identity,
              ortho_init: bool = False, out_gain: float = 1.0) -> nn.Sequential:
    layers = []
    dims = [input_dim, *hidden]
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_d, out_d))
        layers.append(activation())
    layers.append(nn.Linear(dims[-1], output_dim))
    layers.append(out_activation())
    net = nn.Sequential(*layers)

    if ortho_init:
        linears = [m for m in net if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            gain = out_gain if i == len(linears) - 1 else math.sqrt(2)
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)
    return net
