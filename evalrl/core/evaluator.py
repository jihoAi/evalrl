import numpy as np

class Evaluator:
  def __init__(self,eval_env,config):
    self.eval_env = eval_env
    self.config = config

  def evaluate(self,agent) -> dict[str,float]:
    eval_return = np.zeros(self.config.eval_episodes)
    eval_length = np.zeros(self.config.eval_episodes)
    for i in range(self.config.eval_episodes):
      obs, _ = self.eval_env.reset(seed=self.config.eval_seed+i*1000)
      dones = False
      episode_return = 0
      episode_length = 0
      while not dones:
        acts = agent.act(obs,deterministic=True)
        obs,rewards,terms,truncs,infos = self.eval_env.step(acts)
        dones = terms[0] or truncs[0]
        episode_return += rewards[0]
        episode_length+= 1
      eval_return[i] = episode_return
      eval_length[i] = episode_length
    eval_return_mean = eval_return.mean()
    eval_length_mean = eval_length.mean()
    eval_return_std = eval_return.std()
    return {'eval_return_mean':float(eval_return_mean),'eval_return_std':float(eval_return_std),'eval_length_mean':float(eval_length_mean)}
        
      
