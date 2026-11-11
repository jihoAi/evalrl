from math import floor

from evalrl.agents.base import Agent

from evalrl.utils.build_mlp import build_mlp 
from evalrl.utils.replay_buffer import ReplayBuffer

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class DQNAgent(Agent):
  def __init__(self,obs_space,action_space,config,device):
    super().__init__(obs_space,action_space,config,device)

    self.replay_buffer = ReplayBuffer(config.buffer_size,obs_space.shape,device)

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

    self.optimizers['optim'] = optim.Adam(self.networks['q_net'].parameters(),lr=config.learning_rate)

    self.n_owed = 0 #지금까지 이뤄져야했던 업데이트 수 train_freq과 num_envs가 나눠떨어지지않을때 대비
    self.n_done = 0 #지금까지 이뤄진 업데이트

    self.n_target_owed = 0
    self.target_syncs = 0

    self.epsilon = config.eps_start
    self.rng = np.random.default_rng()
  
  def act(self,obs:np.ndarray,deterministic:bool)->np.ndarray:
    self.epsilon = self._epsilon()
    obs = torch.as_tensor(obs,dtype=torch.float32,device=self.device)
    with torch.no_grad():
      q_values = self.networks['q_net'](obs)
    actions = torch.argmax(q_values,dim=1).cpu().numpy()
    if not deterministic:  
      mask = self.rng.random(obs.shape[0])<self.epsilon
      random_actions = self.rng.integers(0,self.action_space.n,size=obs.shape[0])
      actions = np.where(mask,random_actions,actions)
    
    return actions

  def observe(self,obs,next_obs,acts,rewards,terms,truncs):
    self.replay_buffer.add(
      obs=obs,
      actions=acts,
      rewards=rewards,
      next_obs=next_obs,
      terms=terms
    )
  
  def ready_to_update(self):
    self._get_n_owed()
    return self.global_timestep >= self.config.learning_starts and self.n_owed > 0 and len(self.replay_buffer) >= self.config.batch_size
  
  def update(self)->dict[str,float]:
    if self.ready_to_update():
      update_res = {}
      for _ in range(self.n_owed):
        update_res = self._train_step()
        self.n_done +=1
      update_res['n_grad_steps'] = self.n_owed
      self._get_n_target_owed()
      if self.n_target_owed > 0:
        self._sync_target()
        self.target_syncs += self.n_target_owed
      return update_res
    else:
      return None

  def get_state(self)->dict[str,float]:
    return {"epsilon":self.epsilon}
  
  def _get_n_owed(self):
      self.n_owed = floor((self.global_timestep - self.config.learning_starts)/self.config.train_freq) - self.n_done

  def _get_n_target_owed(self):
    self.n_target_owed = floor((self.global_timestep - self.config.learning_starts)/self.config.target_update_freq) - self.target_syncs
  
  def _epsilon(self):
    fraction = self.global_timestep/self.config.eps_decay_steps
    epsilon = max(
      self.config.eps_end,
      self.config.eps_start - fraction*(self.config.eps_start - self.config.eps_end)
    )
    return epsilon

  def _sync_target(self):
    self.networks['target_q'].load_state_dict(self.networks['q_net'].state_dict())

  def _train_step(self)->dict[str,float]:
    batch = self.replay_buffer.sample(self.config.batch_size)
    with torch.no_grad():
      next_q_max = self.networks['target_q'](batch['next_obs']).max(dim=1,keepdim=True).values
      target_q_values = batch['rewards'] + self.config.gamma * next_q_max * (1-batch['terms'])
    q_values = self.networks['q_net'](batch['obs']).gather(1,batch['actions'])
    loss = F.smooth_l1_loss(q_values,target_q_values)
    self.optimizers['optim'].zero_grad()
    loss.backward()
    clip_grad_norm = torch.nn.utils.clip_grad_norm_(self.networks['q_net'].parameters(),self.config.max_grad_norm)
    self.optimizers['optim'].step()
    return {'loss':loss.item(),'q_mean':q_values.mean().item(),'clip_grad_norm':clip_grad_norm.item()}
    
    
    
