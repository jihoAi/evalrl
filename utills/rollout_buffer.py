import numpy as np
import gymnasium as gym
import torch

class RolloutBuffer:
  '''
  정해진 사이즈가 채워지면 업데이트 진행
  '''
  def __init__(self,num_envs:int,num_steps:int,obs_shape:np.ndarray,act_shape:np.ndarray,device):
    self.num_envs = num_envs #환경수
    self.num_steps = num_steps #환경 진행하는 스텝 수
    self.size = num_envs*num_steps #버퍼 사이즈
    self.ptr = 0 #저장 위치

    self.device = device

    self.obs = torch.zeros((self.num_steps,self.num_envs)+self.obs_shape).to(device)
    self.actions = torch.zeros((self.num_steps,self.num_envs)+self.act_shape).to(device)
    self.log_probs = torch.zeros((self.num_steps,self.num_envs)).to(device)
    self.values = torch.zeros((self.num_steps,self.num_envs)).to(device)
    self.rewards = torch.zeros((self.num_steps,self.num_envs)).to(device)
    self.terms = torch.zeros((self.num_steps,self.num_envs)).to(device)
    self.truncs = torch.zeros((self.num_steps,self.num_envs)).to(device)

  def is_full(self)->bool:
    if self.ptr == self.size:
      return True
    else:
      return False

  def add(self,obs,actions,log_probs,values,rewards,terms,truncs):
    self.obs[self.ptr] = obs
    self.actions[self.ptr] = actions
    self.log_probs[self.ptr] = log_probs
    self.values[self.ptr] = values
    self.rewards[self.ptr] = rewards
    self.terms[self.ptr] = terms
    self.truncs[self.ptr] = truncs
    self.ptr+=1
  
  def make_batch(self):
    return {
      'obs' : self.obs.reshape((-1,)+self.obs_shape)),
      'actions : self.actions.reshape((-1,)+self.act_shape)),
      'log_probs : self.log_probs.reshape(-1),
      'values' : self.values.reshape(-1),
      'rewards' : self.rewards.reshape(-1),
      'terms' : self.terms.reshape(-1),
      'truncs' : self.truncs.reshape(-1),
    }

  def clear(self):
    self.ptr = 0
