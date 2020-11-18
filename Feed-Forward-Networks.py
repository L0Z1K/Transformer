import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):
        return  self.model(x)

if __name__ == "__main__":
    FFN = FFN()
    x = torch.rand(512)
    y = FFN(x)

