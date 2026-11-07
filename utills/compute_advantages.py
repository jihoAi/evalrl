import numpy as np
import torch

def compute_advantages(
  rewards:np.ndarray,
  values:np.ndarray,
  next_values:np.ndarray,
  terms:np.ndarray,
  truncs:np.ndarray,
  gamma:float,
  lam:float,
):
    rollout_len = np.arange(rewards.shape[0])
    dones = terms | truncs
    advantages = np.zeros_like(rewards)
    gae = 0
    for t in reversed(rollout_len):
      delta = rewards[t] + gamma*(1-terms[t])*next_values[t] - values[t]
      advantages[t] = gae = delta + gamma*lam*(1-dones[t])*gae
    returns = values + advantages
    return returns, advantages
