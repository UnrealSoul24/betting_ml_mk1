import torch
import torch.nn as nn

class SGPMetaModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6) # 6 Outputs: Prob(PTS), Prob(REB), ...
        )
        
    def forward(self, x):
        return self.net(x)
