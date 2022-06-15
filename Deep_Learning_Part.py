import torch.nn as nn
"""
We brow some deep learning part to help us to handel the large, complex state and action spaces in RL domain.
The success of that method (DL+RL)in gaming, finianianl institiuation and recodmenattaion system, clude and system already successfull.   
Deep Learning Part (DL)
NN Network class for Agent (Brain/calculation of input and wights) 
Done
"""
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
    def forward(self, x):

        return self.fc(x)
