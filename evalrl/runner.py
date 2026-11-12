import os
import random 
import shutil
import csv

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

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def run_one(algo, seed, overrides=None, config_dir="configs/cartpole",runs_root="runs",device="cpu"):
  train_env=None
  eval_env=None
  try:
    run_dir = os.path.join(runs_root,f"{algo}_seed{seed}")
    if os.path.exists(os.path.join(run_dir,"final.pt")):
      print(f"{algo} seed {seed} exist. skip")
      with open(os.path.join(run_dir,"eval.csv"), newline="") as f:
        row = list(csv.DictReader(f))[-1]
        result = {metric:float(value) for metric, value in row.items() if metric !="step"}
      return result
    else:
      if os.path.exists(run_dir):
        print(f"기존에 존재하던 {run_dir}을 삭제하고 다시 학습을 진행합니다.")
        shutil.rmtree(run_dir)
      if overrides==None:
        overrides={}
  
      config = load_config(algo=algo,overrides={**overrides,"seed":seed,"run_dir":run_dir},config_dir=config_dir)
      set_seed(seed)
  
      train_env = make_train_envs(config)
      eval_env = make_eval_env(config)
      
      agent = AGENT_REGISTRY[algo](
        obs_space = train_env.single_observation_space,
        action_space = train_env.single_action_space,
        config = config,
        device = device
      )
      
      evaluator = Evaluator(eval_env,config)
      
      logger = Logger(run_dir=run_dir)
      
      trainer = Trainer(
        agent = agent,
        envs = train_env,
        config = config,
        evaluator = evaluator,
        logger = logger
      )
      
      result = trainer.train()
      return result
  finally:
    if train_env is not None:
      train_env.close()
    if eval_env is not None:
      eval_env.close()
    
def run_all(algos, seeds, overrides=None,config_dir="configs/cartpole",runs_root="runs",device="cpu"):
  result = {}
  num_iter = len(algos)*len(seeds)
  iter = 1
  for seed in seeds:
    for algo in algos:
      try:
        result[(algo,seed)] = run_one(
          algo=algo,
          seed=seed,
          overrides=overrides,
          config_dir=config_dir,
          runs_root=runs_root,
          device=device
        )
        print(f"[{iter}/{num_iter}] {algo} seed {seed} 완료")
        iter+=1
      except Exception as e:
        print(f"[{algo}:{seed}] {type(e).__name__}: {e}")
  return result
