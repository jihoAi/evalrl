import torch.nn as nn

def build_mlp(input_dim: int, output_dim: int, hidden: tuple[int, ...],
              activation=nn.ReLU, out_activation=nn.Identity) -> nn.Sequential:
    layers = []
    dims = [input_dim, *hidden]
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_d, out_d))
        layers.append(activation())
    layers.append(nn.Linear(dims[-1], output_dim))
    layers.append(out_activation())
    return nn.Sequential(*layers)
