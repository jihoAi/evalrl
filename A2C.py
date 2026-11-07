from Agent import Agent

from utils.build_mlp import build_mlp 
from utils.rollout_buffer import RolloutBuffer
from utils.compute_advantages import compute_advantages

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
      self._last_log_prob = dist.log_prob(actions).cpu().numpy()
      self._last_value = self.networks['critic'](obs).squeeze(-1).cpu().numpy()
    return actions.cpu().numpy()

  def observe(self, obs: np.ndarray, next_obs: np.ndarray,acts: np.ndarray, rewards: np.ndarray, terms: np.ndarray, truncs: np.ndarray):
    #환경은 samestep사용한다고 가정

    self.rollout_buffer.add(obs,acts,self._last_log_prob,self._last_value,rewards,terms,truncs)
    self._last_log_prob = None
    self._last_value = None

  def ready_to_update(self)->bool:
    return self.rollout_buffer.is_full()

  def update(self)->dict[str,float]:
    if self.ready_to_update():
      with torch.no_grad():
        next_obs_tensor = self.rollout_buffer.next_obs_tensor()
        next_values = self.networks['critic'](next_obs_tensor).cpu().numpy()
      returns,advantages = compute_advantages(
        self.rollout_buffer.rewards,
        self.rollout_buffer.values,
        next_values,
        self.rollout_buffer.terms,
        self.rollout_buffer.truncs,
        self.config.gamma,
        self.config.gae_lambda
      )
      self.rollout_buffer.set_advantages(advantages,returns)

    batches = self.rollout_buffer.make_batch()
    logits = self.networks['actor'](batches['obs'])
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(batches['actions'])
    entropies = dist.entropy().mean()
    values = self.networks['critic'](batches['obs']).squeeze(-1)

    policy_loss = log_probs*batches['advantages'].mean()

    value_loss = ((batches['returns']-values)**2).mean()

    loss = -policy_loss + self.config.vf_coef*value_loss - self.config.ent_coef*entropies

    self.optimizers['optim'].zero_grad()
    self.optimizers['optim'].backward()
    self.optimizers['optim'].step()

    self.rollout_buffer.clear()

    return {'policy_loss':policy_loss,'value_loss':value_loss,'entropy':entropies}
    

    
