import yaml
import os
from dataclasses import asdict

from evalrl.configs import A2CConfig, PPOConfig, DQNConfig

def load_yaml(path):
  with open(path,encoding="utf-8") as f:
    config = yaml.safe_load(f)
  if config:
    return config
  else:
    return {}

def deep_merge(base,override):
  result = dict(base)
  for key, value in override.items():
    if key in result and isinstance(result[key],dict) and isinstance(value,dict):
      result[key] = deep_merge(result[key],value)
    else:
      result[key]=value
  return result

def load_config(algo, overrides=None, config_dir="configs"):
  CONFIG_REGISTRY = {
    "ppo" : PPOConfig,
    "a2c" : A2CConfig,
    "dqn" : DQNConfig
  }
  
  if algo not in CONFIG_REGISTRY:
    raise ValueError(f"알 수 없는 algo: {algo!r}. 사용 가능: {list(CONFIG_REGISTRY)}")

  cls = CONFIG_REGISTRY[algo]
  
  common = load_yaml(os.path.join(config_dir,"common.yaml"))
  algo_yaml = load_yaml(os.path.join(config_dir,f"{algo}.yaml"))

  overrides = overrides or {}

  result = deep_merge(common,algo_yaml)
  result = deep_merge(result,overrides)

  return cls(**result)
  


def save_config(config,path):
  data = asdict(config)
  data["hidden"] = list(data["hidden"])
  folder = os.path.dirname(path)
  if folder:
    os.makedirs(os.path.dirname(path),exist_ok=True)
  with open(path,"w",encoding="utf-8") as f:
    yaml.safe_dump(data,f,allow_unicode=True,sort_keys=False)
