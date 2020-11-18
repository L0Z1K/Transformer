import torch
import torch.nn as nn
from math import sqrt

class Attention(nn.Module):
    def __init__(self, dk=64):
        super().__init__()

        self.MatMul = torch.matmul
        self.Scale = (lambda x: x*1/sqrt(dk))
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, Q, K, V):
        out = self.MatMul(Q, K.t())
        out = self.Scale(out)
        out = self.Softmax(out)
        out = self.MatMul(out, V)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8):
        super().__init__()
        self.Attention = Attention()
        self.W_Q = nn.Linear(d_model, h*d_k)
        self.W_K = nn.Linear(d_model, h*d_k)
        self.W_V = nn.Linear(d_model, h*d_v)
        self.W_O = nn.Linear(h*d_v, d_model)

    def forward(self, Q, K, V):
        return self.W_O(self.Attention(self.W_Q(Q), self.W_K(K), self.W_V(V)))

if __name__ == "__main__":
    MHA = MultiHeadAttention()

    Q = torch.ones(3, 512)
    K = torch.ones(4, 512)
    V = torch.ones(4, 512)
    output = MHA(Q, K, V)
    print(output.shape)        