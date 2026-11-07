import numpy as np
import torch


class RolloutBuffer:
    '''
    정해진 스텝이 채워지면 업데이트 진행.
    '''
    def __init__(self, num_envs: int, num_steps: int, obs_shape: tuple, act_shape: tuple, device, act_dtype=np.int64):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.size = num_envs * num_steps
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.device = device
        self.ptr = 0

        shape = (num_steps, num_envs)
        self.obs = np.zeros(shape + obs_shape, dtype=np.float32)
        self.actions = np.zeros(shape + act_shape, dtype=act_dtype)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.next_values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.terms = np.zeros(shape, dtype=bool)
        self.truncs = np.zeros(shape, dtype=bool)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)

    def is_full(self) -> bool:
        return self.ptr == self.num_steps

    def add(self, obs, actions, log_probs, values, next_values, rewards, terms, truncs):
        if self.is_full():
            raise RuntimeError("RolloutBuffer is full. Call clear() after update.")
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.next_values[self.ptr] = next_values
        self.rewards[self.ptr] = rewards
        self.terms[self.ptr] = terms
        self.truncs[self.ptr] = truncs
        self.ptr += 1

    def set_advantages(self, advantages: np.ndarray, returns: np.ndarray):
        self.advantages[:] = advantages
        self.returns[:] = returns

    def make_batch(self) -> dict[str, torch.Tensor]:
        def to_tensor(x: np.ndarray, tail: tuple = ()) -> torch.Tensor:
            return torch.as_tensor(x.reshape((-1,) + tail), device=self.device)

        return {
            'obs': to_tensor(self.obs, self.obs_shape),
            'actions': to_tensor(self.actions, self.act_shape),
            'log_probs': to_tensor(self.log_probs),
            'values': to_tensor(self.values),
            'advantages': to_tensor(self.advantages),
            'returns': to_tensor(self.returns),
        }

    def clear(self):
        self.ptr = 0
