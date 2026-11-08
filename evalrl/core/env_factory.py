import gymnasium as gym

def _make(config,num_envs):
  envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make(config.env_id) for _ in range(num_envs)],
    autoreset_mode = gym.vector.AutoresetMode.SAME_STEP,
  )
  return envs
  
def make_train_envs(config):
  return _make(config,config.num_envs)

def make_eval_env(config):
  return _make(config,config.eval_episodes)
  
