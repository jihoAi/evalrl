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

    self._last_log_prob = None
    self._last_value = None

  def act(self,obs:np.ndarray,deterministic:bool):
    obs = torch.as_tensor(obs,dtype=torch.float32).to(self.device)
    logits = self.networks['actor'](obs)
    dist = Categorical(logits=logtis)
    actions = logits.argmax(-1) if deterministic else dist.sample()
    self._last_log_prob = dist.log_prob(action)
    with torch.no_gard()
      self._last_value = self.networks['critic'](obs)
    return actions

  def observe(self, obs: np.ndarray, acts: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, terms: np.ndarray, truncs: np.ndarray):
    #환경은 samestep사용한다고 가정
    obs = torch.as_tensor(obs,dtype=torch.float32,device=self.device)
    next_obs = torch.as_tensor(next_obs,dtype=torch.float32,device=self.device)
    acts = torch.as_tensor(acts,dtype=torch.long,devcie=self.device)
    rewards = torch.as_tensor(acts,dtype=torch.float32,device=self.device)
    terms = torch.as_tensor(terms,dtype=torch.float32,device=self.device)
    truncs = torch.as_tensor(truncs,dtype=torch.float32,device=self.device)

    self.rollout_buffer.add(obs,next_obs,acts,self._last_log_prob,self._last_value,rewards,terms,truncs)
    self._last_log_prob = None
    self._last_value = None

  
    
    
