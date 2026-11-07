import torch

class RolloutBuffer:
    '''
    정해진 스텝이 채워지면 업데이트 진행
    '''
    def __init__(self, num_envs: int, num_steps: int, obs_shape: tuple, act_shape: tuple, device, act_dtype=torch.long):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.size = num_envs * num_steps
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.device = device
        self.ptr = 0

        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.next_obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + act_shape, dtype=act_dtype, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.terms = torch.zeros((num_steps, num_envs), device=device)
        self.truncs = torch.zeros((num_steps, num_envs), device=device)

    def is_full(self) -> bool:
        return self.ptr == self.num_steps

    def add(self, obs, next_obs,actions, log_probs, values, rewards, terms, truncs):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.rewards[self.ptr] = rewards
        self.terms[self.ptr] = terms
        self.truncs[self.ptr] = truncs
        self.ptr += 1

    def make_batch(self) -> dict[str, torch.Tensor]:
        return {
            'obs': self.obs.reshape((-1,) + self.obs_shape),
            'next_obs' : self.next_obs.reshape((-1,) + self.obs_shape),
            'actions': self.actions.reshape((-1,) + self.act_shape),
            'log_probs': self.log_probs.reshape(-1),
            'values': self.values.reshape(-1),
        }

    def clear(self):
        self.ptr = 0
