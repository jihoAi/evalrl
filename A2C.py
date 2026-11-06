import Agent from Agent

import build_mlp from utils.build_mlp
import RolloutBuffer from utils.rollout_buffer

import torch
import torch.optim as optim
from torch.distributions import Categorical

class A2C(Agent):
  def __init__(self,obs_space,action_space,config,device):
    super().__init__(obs_space,action_space,config,device):

    self.rollout_buffer = RolloutBuffer(
      num_envs=config['num_envs'],
      num_steps=config['num_steps'],
      obs_shape=obs_space.single_observation_space.shape,
      act_shape=action_space.single_action_space.shape

    #Actor, Critic, optimizer 정의
    self.netowrks['actor'] = build_mlp(
      input_dim = obs_space.single_observation_space.shape,
      output_dim = action_space.single_observation_space.n,
      hidden = config['hidden']
    )

    self.netowrks['critic'] = build_mlp(
      input_dim = obs_space.single_observation_space.shape,
      output_dim = action_space.single_observation_space.n,
      hidden = config['hidden']
    )

    self.optimizers['actor_optim'] = optim.Adam(
      self.networks['actor'].parameters(),
      lr=config['learning_rate']
    )

    self.optimizers['critic_optim'] = optim.Adam(
      self.networks['actor'].parameters(),
      lr=config['learning_rate']
    )

def act(self,obs:np.ndarray,deterministic:bool):
  obs = torch.as_tensor(obs,dtype=torch.float32).to(self.device)
  logits = self.networks['actor'](obs)
  dist = Categorical(logits=logtis)
  actions = logits.argmax(-1) if deterministic else dist.sample()
  values = self.networks['critic'](obs)
  return actions, dist.log_prob(action), dist.entropy(), values

def observe(self,obs:torch.Tensor,acts:torch.Tensor,rewards:torch.Tensor,
            next_obs:torch.Tensor,terms:torch.Tensor,truncs:torch.Tensor):
  
  
