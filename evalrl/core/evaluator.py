import numpy as np

class Evaluator:
  def __init__(self,eval_env,config):
    self.eval_env = eval_env
    self.config = config

  def evaluate(self,agent) -> dict[str,float]:
    finished = np.zeros(self.config.eval_episodes,dtype=bool)
    eval_return = np.zeros(self.config.eval_episodes)
    eval_length = np.zeros(self.config.eval_episodes)
    obs, _ = self.eval_env.reset(seed=self.config.eval_seed)
    while not finished.all():
      acts = agent.act(obs,deterministic=True)
      obs,rewards,terms,truncs,infos = self.eval_env.step(acts)
      dones = terms | truncs
      newly_dones = dones & ~finished
      eval_return[~finished] += rewards[~finished]
      eval_length[~finished] += 1
      finished |= newly_dones
    eval_return_mean = eval_return.mean()
    eval_length_mean = eval_length.mean()
    eval_return_std = eval_return.std()
    return {'eval_return_mean':float(eval_return_mean),'eval_return_std':float(eval_return_std),'eval_length_mean':float(eval_length_mean)}
        
      
