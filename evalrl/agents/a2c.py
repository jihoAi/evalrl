from evalrl.agents.base import Agent

from evalrl.utils.build_mlp import build_mlp 
from evalrl.utils.rollout_buffer import RolloutBuffer
from evalrl.utils.compute_advantages import compute_advantages

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

ACTIVATION_MAP = {"relu":nn.ReLU,"tanh":nn.Tanh}

class A2CAgent(Agent):
  def __init__(self,obs_space,action_space,config,device):
    super().__init__(obs_space,action_space,config,device)
    activation = ACTIVATION_MAP[config.activation]
    self.rollout_buffer = RolloutBuffer(
      num_envs=config.num_envs,
      num_steps=config.num_steps,
      obs_shape=obs_space.shape,
      act_shape=action_space.shape,
      device = device
    )

    #Actor, Critic, optimizer 정의
    self.networks['actor'] = build_mlp(
      input_dim = obs_space.shape[0],
      output_dim = action_space.n,
      hidden = config.hidden,
      activation = activation,
      ortho_init = config.ortho_init,
      out_gain = 0.01
    ).to(device)

    self.networks['critic'] = build_mlp(
      input_dim = obs_space.shape[0],
      output_dim = 1,
      hidden = config.hidden,
      activation = activation,
      ortho_init = config.ortho_init,
      out_gain = 1.0
    ).to(device)

    self.optimizers['actor'] = optim.Adam(self.networks['actor'].parameters(), lr=config.actor_lr)
    self.optimizers['critic'] = optim.Adam(self.networks['critic'].parameters(), lr=config.critic_lr)
    
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
    self.rollout_buffer.add(
      obs=obs,
      next_obs=next_obs,
      actions=acts,
      log_probs=self._last_log_prob,
      values=self._last_value,
      rewards=rewards,
      terms=terms,
      truncs=truncs
    )
    self._last_log_prob = None
    self._last_value = None

  def ready_to_update(self)->bool:
    return self.rollout_buffer.is_full()
    
  def _update_lrs(self):
    self._apply_lr_schedule(self.optimizers['actor'], self.config.actor_lr)
    self._apply_lr_schedule(self.optimizers['critic'], self.config.critic_lr)

  def get_state(self) -> dict[str, float]:
    return {
      'actor_lr': self.optimizers['actor'].param_groups[0]['lr'],
      'critic_lr': self.optimizers['critic'].param_groups[0]['lr'],
    }

  def update(self)->dict[str,float]:
    if self.ready_to_update():
      self._update_lrs()
      self._set_returns_and_adv()

      batches = self.rollout_buffer.make_batch()
      logits = self.networks['actor'](batches['obs'])
      dist = Categorical(logits=logits)
      log_probs = dist.log_prob(batches['actions'])
      entropies = dist.entropy().mean()
      values = self.networks['critic'](batches['obs']).squeeze(-1)

      policy_loss = -(log_probs*batches['advantages']).mean()
      value_loss = 0.5*((batches['returns']-values)**2).mean()
      actor_loss = policy_loss - self.config.ent_coef*entropies

      self.optimizers['actor'].zero_grad()
      actor_loss.backward()
      actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.networks['actor'].parameters(),max_norm=self.config.max_grad_norm)
      self.optimizers['actor'].step()

      self.optimizers['critic'].zero_grad()
      value_loss.backward()
      critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.networks['critic'].parameters(),max_norm=self.config.max_grad_norm)
      self.optimizers['critic'].step()

      self.rollout_buffer.clear()

      return {'policy_loss':policy_loss.item(),'value_loss':value_loss.item(),'entropy':entropies.item(),
              'actor_grad_norm':actor_grad_norm.item(),'critic_grad_norm':critic_grad_norm.item()}
    else:
      return None
      
  def _set_returns_and_adv(self):
    with torch.no_grad():
        next_obs_tensor = self.rollout_buffer.next_obs_tensor()
        next_values = self.networks['critic'](next_obs_tensor).cpu().numpy().squeeze(-1)
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
