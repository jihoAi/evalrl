import abc
import numpy as np
import torch

class Agent(abc.ABC):
  def __init__(self,obs_space,action_space,config,device):
    self.obs_space = obs_space
    self.action_space = action_space
    self.config = config
    self.device = device
    self.global_timestep = 0
    self.networks = {} #네트워크
    self.optimizers = {} #옵티마이저

  # 관찰을 받고 행동을 반환한다. 학습이라면 deterministic=False 평가라면 True
  @abc.abstractmethod
  def act(self,obs:np.ndarray,deterministic:bool)->np.ndarray:
    pass

  #transition을 버퍼에 저장
  @abc.abstractmethod
  def observe(self, obs: np.ndarray, next_obs: np.ndarray, acts: np.ndarray, rewards: np.ndarray, terms: np.ndarray, truncs: np.ndarray):
    pass

  @abc.abstractmethod
  def ready_to_update(self)->bool:
    pass

  @abc.abstractmethod
  def update(self)->dict[str,float]:
    pass

  def save(self,path:str):
    saved_params = {
      'networks' : {name:network.state_dict() for name,network in self.networks.items()},
      'optimizers' : {name:optim.state_dict() for name,optim in self.optimizers.items()},
      'global_timestep' : self.global_timestep
    }
    torch.save(saved_params,path)

  def load(self,path:str):
    ckpt = torch.load(path,weights_only=True,map_location = self.device)
    for name,network in self.networks.items():
      network.load_state_dict(ckpt['networks'][name])
    for name,optim in self.optimizers.items():
      optim.load_state_dict(ckpt['optimizers'][name])
    self.global_timestep = ckpt['global_timestep']
    
  
  def on_step(self,global_timestep):
    self.global_timestep = global_timestep

  def get_state(self)->dict[str,float]:
    return dict()

  def on_episode_end(self,info):
    pass

  def _apply_lr_schedule(self, optimizer, base_lr: float) -> float:
    if self.config.lr_schedule == "linear":
      frac = max(1.0 - self.global_timestep / self.config.total_timestep, 0.0)
      lr = base_lr * frac
    else:
      lr = base_lr
    for g in optimizer.param_groups:
      g['lr'] = lr
    return lr
