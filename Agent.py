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
  def observe(self,obs:np.ndarray,acts:np.ndarray,rewards:np.ndarray,next_obs:np.ndarray,terms:np.ndarray,truncs:np.ndarray):
    pass

  @abc.abstractmethod
  def ready_to_update(self)->bool:
    pass

  @abc.abstractmethod
  def update(self)->dict[str,float]:
    pass

  def save(self,path:str):
    saved_params = {}
    for name, network in self.networks:
      saved_params[name] = network.state_dict()
    for name, optimizer in self.optimizers:
      saved_params[name] = optimizer.state_dict()
    saved_params['global_timestep'] = self.global_timestep
    torch.save(saved_params,path,map_location = self.device)

  def load(self,path:str):
    ckpt = torch.load(path,weights_only=True)
    self.networks = ckpt['networks']
    self.optimizers = ckpt['optimizers']
    self.global_timestep = ckpt['global_timestep']
    
  
  def on_step(self,global_timestep):
    self.global_timestep = global_timestep
    pass

  def get_state(self)->dict[str,float]:
    pass

  def on_episode_end(self,info):
    pass
