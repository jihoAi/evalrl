import os
import random 
import shutil

import numpy as np
import torch

from evalrl.config_loader import load_config
from evalrl.agents.a2c import A2CAgent
from evalrl.agents.ppo import PPOAgent
from evalrl.agents.dqn import DQNAgent
from evalrl.core.env_factory import make_train_envs, make_eval_env
from evalrl.core.evaluator import Evaluator
from evalrl.core.logger import Logger
from evalrl.core.trainer import Trainer

AGENT_REGISTRY = {
  'a2c' : A2CAgent,
  'dqn' : DQNAgent,
  'ppo' : PPOAgent
}

def set_seed(seed,torch_deterministic):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = torch_deterministic

def run_one(algo, seed, overrides=None, config_dir="configs"):
  pass

def run_all(algos, seeds, overrides=None):
  pass
