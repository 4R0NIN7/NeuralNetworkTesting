import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim,device) -> None:
        super().__init__()
        # define different layers
        self.lin = nn.Linear(input_dim, output_dim,device=device)  # only one layer

    def forward(self, x):
        return self.lin(x)
