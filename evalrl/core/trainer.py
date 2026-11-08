import os

import numpy as np
import time


class Trainer:
  def __init__(self,agent,envs,config,evaluator,logger):
    self.agent = agent
    self.envs = envs
    self.evaluator = evaluator
    self.logger = logger
    self.config = config
    
    self.ep_returns = np.zeros(config.num_envs)
    self.ep_length = np.zeros(config.num_envs)
    self.finished = []
    self.length = []
    self.last_update_metrics = None
    self.last_log = agent.global_timestep #가장 최근 로그 step
    self.last_eval = agent.global_timestep #가장 최근 평가 step
    self.last_save = agent.global_timestep #가장 최근 저장 step

    self.start_time = None
    self.last_log_time = None
  

  def train(self):
    step = self.agent.global_timestep
    self.start_time = time.time()
    self.last_log_time = time.time()
    obs,_ = self.envs.reset(seed=self.config.seed)
    self._evaluate(step)
    try:
      while step < self.config.total_timestep:
        acts = self.agent.act(obs,deterministic=False)
        next_obs, rewards, terms, truncs, infos = self.envs.step(acts)

        real_next_obs = self._real_next_obs(next_obs,infos)

        self.agent.observe(
          obs=obs,
          next_obs=real_next_obs,
          acts=acts,
          rewards=rewards,
          terms=terms,
          truncs=truncs
        )
        dones = terms | truncs
        step += self.config.num_envs
        self.agent.on_step(step)

        self._record_episodes(rewards,dones)

        if self.agent.ready_to_update():
          self.last_update_metrics = self.agent.update()

        if step - self.last_log >= self.config.log_interval and self.last_update_metrics is not None:
          self._log_train(step)

        if step - self.last_eval >= self.config.eval_interval:
          self.last_eval = step
          self._evaluate(step)

        if step- self.last_save >= self.config.save_interval:
          self.last_save = step
          self._save(step)
        
        obs = next_obs
      result = self._evaluate(step)
      self._save(step,final=True)
      return result
    finally:
      self.logger.close()

  def _real_next_obs(self,next_obs,infos)->np.ndarray:
    real_next_obs = next_obs.copy()
    if "_final_obs" in infos:
      mask = infos["_final_obs"]
      real_next_obs[mask] = np.stack(infos["final_obs"][mask])
    return real_next_obs

  def _record_episodes(self,rewards,dones):
    self.ep_returns += rewards
    self.ep_length += 1
    self.finished.extend(self.ep_returns[dones].tolist())
    self.length.extend(self.ep_length[dones].tolist())
    self.ep_returns[dones] = 0
    self.ep_length[dones] = 0

  def _log_train(self, step):
    now = time.time()
    metrics = {
        "episode_return_mean": float(np.mean(self.finished[-20:])) if self.finished else float("nan"),
        "episode_length_mean": float(np.mean(self.length[-20:])) if self.length else float("nan"),
        "episodes": len(self.finished),
        "sps": (step - self.last_log) / max(now - self.last_log_time, 1e-8),
        "time_elapsed": now - self.start_time,
        **self.last_update_metrics,
        **self.agent.get_state(),
    }
    self.logger.log("train", step, metrics)
    self.last_log = step
    self.last_log_time = now
    
  def _evaluate(self, step):
    result = self.evaluator.evaluate(self.agent)
    self.logger.log("eval", step, result)
    return result

  def _save(self, step, final=False):
    name = "final.pt" if final else "latest.pt"
    self.agent.save(os.path.join(self.config.run_dir, name))
