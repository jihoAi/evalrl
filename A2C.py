from Agent import Agent

from utils.build_mlp import build_mlp 
from utils.rollout_buffer import RolloutBuffer 

import torch
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

class A2CAgent(Agent):
  def __init__(self,obs_space,action_space,config,device):
    super().__init__(obs_space,action_space,config,device):

    self.rollout_buffer = RolloutBuffer(
      num_envs=config.num_envs,
      num_steps=config.num_steps,
      obs_shape=obs_space.single_observation_space.shape,
      act_shape=action_space.single_action_space.shape,
      device = device
    )

    #Actor, Critic, optimizer 정의
    self.networks['actor'] = build_mlp(
      input_dim = obs_space.shape[0],
      output_dim = action_space.n,
      hidden = config.hidden
    ).to(device)

    self.networks['critic'] = build_mlp(
      input_dim = obs_space.shape[0],
      output_dim = 1,
      hidden = config.hidden
    ).to(device)

    self.optimizers['optim'] = optim.Adam([
        {'params':self.networks['actor']},
        {'params':self.networks['critic']}
      ],lr=config.learning_rate
    )

    self._last_log_prob = None
    self._last_value = None

  def act(self,obs:np.ndarray,deterministic:bool)->np.ndarray:
    obs = torch.as_tensor(obs,dtype=torch.float32).to(self.device)
    with torch.no_grad():
      logits = self.networks['actor'](obs)
      dist = Categorical(logits=logits)
      actions = logits.argmax(-1) if deterministic else dist.sample()
      self._last_log_prob = dist.log_prob(actions)
      self._last_value = self.networks['critic'](obs).squeeze(-1)
    return actions.cpu().numpy()

  def observe(self, obs: np.ndarray, acts: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, terms: np.ndarray, truncs: np.ndarray):
    #환경은 samestep사용한다고 가정
    obs = torch.as_tensor(obs,dtype=torch.float32,device=self.device)
    next_obs = torch.as_tensor(next_obs,dtype=torch.float32,device=self.device)
    acts = torch.as_tensor(acts,dtype=torch.long,device=self.device)
    rewards = torch.as_tensor(rewards,dtype=torch.float32,device=self.device)
    terms = torch.as_tensor(terms,dtype=torch.bool,device=self.device)
    truncs = torch.as_tensor(truncs,dtype=torch.bool,device=self.device)

    self.rollout_buffer.add(obs,next_obs,acts,self._last_log_prob,self._last_value,rewards,terms,truncs)
    self._last_log_prob = None
    self._last_value = None

  def ready_to_update(self)->bool:
    return self.rollout_buffer.is_full()

  def update(self)->dict[str,float]:
  
    
    
