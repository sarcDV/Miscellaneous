import torch
import torch.nn as nn

class HMM(nn.Module):
    def __init__(self, num_states, num_obs):
        super(HMM, self).__init__()
        self.num_states = num_states
        self.num_obs = num_obs
        self.A = nn.Parameter(torch.randn(num_states, num_states))
        self.B = nn.Parameter(torch.randn(num_states, num_obs))
        self.pi = nn.Parameter(torch.randn(num_states))

    def forward(self, obs):
        alpha = self.pi * self.B[:, obs[0]]
        for t in range(1, len(obs)):
            alpha = torch.matmul(alpha.unsqueeze(0), self.A) * self.B[:, obs[t]]
        return alpha.sum()

# Esempio di utilizzo
model = HMM(num_states=3, num_obs=2)
obs = torch.tensor([0, 1, 0, 1])
log_prob = model(obs)
print(log_prob)
