from evalrl.agents.base import Agent

from evalrl.utils.build_mlp import build_mlp 
from evalrl.utils.replay_buffer import ReplayBuffer

import torch
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

class DQNAgent(Agent):
  def __init__(self,obs_space,action_space,config,device):
    super().__init__(obs_space,action_space,config,device)

    self.replay_buffer = ReplayBuffer(config.replay_size,obs_space.shape,device)

    self.networks['q_net'] = build_mlp(
      input_dim = obs_space.shape[0],
      output_dim = action_space.n,
      hidden = config.hidden
    ).to(device)

    self.networks['target_q']  = build_mlp(
      input_dim = obs_space.shape[0],
      output_dim = action_space.n,
      hidden = config.hidden
    ).to(device)

    self.networks['target_q'].load_state_dict(self.networks['q_net'].state_dict())

    self.optimizers['optim'] = optim.Adam(self.networks['q_net'].parameters(),lr=config.lr)

    
    
    
