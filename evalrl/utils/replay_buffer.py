import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device

        self.obs_buffer = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.next_obs_buffer = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.actions_buffer = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.terms_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, obs, actions, rewards, next_obs, terms):
        obs = np.asarray(obs, dtype=np.float32).reshape((-1,) + self.obs_shape)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape((-1,) + self.obs_shape)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1, 1)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        terms = np.asarray(terms, dtype=np.float32).reshape(-1, 1)

        n = len(obs)
        if n > self.capacity:
            obs, actions, rewards, next_obs, terms = (
                x[-self.capacity:] for x in (obs, actions, rewards, next_obs, terms)
            )
            n = self.capacity

        idx = (self.ptr + np.arange(n)) % self.capacity

        self.obs_buffer[idx] = obs
        self.actions_buffer[idx] = actions
        self.rewards_buffer[idx] = rewards
        self.next_obs_buffer[idx] = next_obs
        self.terms_buffer[idx] = terms

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs_buffer[idx], device=self.device),
            "actions": torch.as_tensor(self.actions_buffer[idx], device=self.device),
            "rewards": torch.as_tensor(self.rewards_buffer[idx], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs_buffer[idx], device=self.device),
            "terms": torch.as_tensor(self.terms_buffer[idx], device=self.device),
        }

    def is_full(self)->bool:
        return self.size == self.capacity
    
    def __len__(self):
        return self.size
