# evalrl

여러 강화학습 에이전트를 **같은 조건에서** 구현·실행·비교하기 위한 PyTorch 라이브러리입니다.

직접 구현한 DQN · A2C · PPO를 하나의 공통 프레임워크에서 학습·평가해 공정하게 비교하는 것이 목표입니다. 원칙은 **알고리즘만 바꾸고 나머지는 모두 고정**하는 것입니다.

## 현재 상태

| 구성 요소 | 상태 |
|---|---|
| 베이스 `Agent` (save / load / on_step) | ✅ 완료 |
| `A2CAgent` | ✅ 완료 (CartPole 학습 확인) |
| `PPOAgent` (`A2CAgent` 상속) | ✅ 완료 (CartPole 500 도달) |
| Rollout buffer, GAE | ✅ 완료 |
| Config (dataclass, `configs.py`) | ✅ 완료 |
| env factory (SAME_STEP vector env) | ✅ 완료 |
| Evaluator | ✅ 완료 (테스트 통과) |
| Logger (CSV) | ✅ 완료 (테스트 통과) |
| Trainer | 🚧 진행 중 |
| `DQNAgent` | ⏳ 예정 |
| Runner (algo × seed) · 분석 스크립트 | ⏳ 예정 |

## 공정성 기준

| 항목 | 통일 방법 |
|---|---|
| 학습 예산 | 환경 step 수 기준 |
| 시드 | 같은 seed 집합, 알고리즘당 5~10개 |
| 네트워크 | 같은 구조, 공통 팩토리 `build_mlp`로 생성 |
| 환경 | 같은 래퍼 · `max_episode_steps`, SAME_STEP autoreset |
| 종료 처리 | truncation에서는 bootstrap, termination에서는 안 함 (세 알고리즘 동일) |
| 평가 | 별도 eval env, 고정 eval seed, deterministic policy |

환경은 DQN이 이산 행동(discrete action)만 지원하므로 CartPole · LunarLander를 사용합니다.

## 디렉토리 구조

```
evalrl/                     (저장소 루트)
├── pyproject.toml
├── README.md
├── tests/
│   └── test_logger_evaluator.py
└── evalrl/                 (패키지)
    ├── __init__.py
    ├── configs.py          CommonConfig, A2CConfig, PPOConfig
    ├── agents/
    │   ├── __init__.py
    │   ├── base.py         Agent (추상 클래스)
    │   ├── a2c.py          A2CAgent
    │   └── ppo.py          PPOAgent
    ├── utils/
    │   ├── __init__.py
    │   ├── build_mlp.py
    │   ├── rollout_buffer.py
    │   └── compute_advantages.py
    └── core/
        ├── __init__.py
        ├── env_factory.py  make_train_envs, make_eval_env
        ├── evaluator.py    Evaluator
        ├── logger.py       Logger
        └── trainer.py      (작성 중)
```

## 설치

```bash
git clone https://github.com/jihoAi/evalrl.git
pip install -e evalrl
```

`-e`(editable) 설치라 코드를 고치면 재설치 없이 반영됩니다.

### Google Colab

노트북 첫 셀:

```python
import os
REPO = "/content/repos/evalrl"   # /content 바로 아래에 두면 이름이 겹쳐 import가 꼬임
if os.path.exists(REPO):
    !git -C {REPO} pull
else:
    !git clone https://github.com/jihoAi/evalrl.git {REPO}
!pip install -q -e {REPO}
```

설치 후 **런타임 → 세션 다시 시작**을 한 번 해야 import됩니다.

코드를 고친 뒤에는 GitHub에 push → 위 셀(pull) 실행 → 세션 다시 시작 순서로 반영합니다. Colab(Python 3.12+)에서는 `%autoreload`가 `imp` 모듈 오류로 동작하지 않으므로 쓰지 않습니다. `!python ...`으로 실행하는 스크립트는 매번 새 프로세스라 재시작 없이 최신 코드가 반영됩니다.

> Colab에서 파일을 직접 고치면 다음 `git pull`이 거부될 수 있습니다. 수정은 GitHub 쪽에서만 하고 Colab은 pull만 하세요.

## 빠른 시작

```python
import numpy as np
import torch

from evalrl.agents import PPOAgent
from evalrl.configs import PPOConfig
from evalrl.core.env_factory import make_train_envs, make_eval_env
from evalrl.core.evaluator import Evaluator

config = PPOConfig(total_timestep=200_000)
device = torch.device("cpu")

envs = make_train_envs(config)   # SAME_STEP vector env, config.num_envs개
agent = PPOAgent(envs.single_observation_space, envs.single_action_space, config, device)

obs, _ = envs.reset(seed=config.seed)
global_timestep = 0
while global_timestep < config.total_timestep:
    acts = agent.act(obs, deterministic=False)
    next_obs, rewards, terms, truncs, infos = envs.step(acts)

    # SAME_STEP: 끝난 env는 진짜 마지막 obs로 교체해서 observe에 넘긴다
    real_next_obs = next_obs.copy()
    if "_final_obs" in infos:
        mask = infos["_final_obs"]
        real_next_obs[mask] = np.stack(infos["final_obs"][mask])

    agent.observe(obs=obs, next_obs=real_next_obs, acts=acts,
                  rewards=rewards, terms=terms, truncs=truncs)

    global_timestep += config.num_envs
    agent.on_step(global_timestep)

    if agent.ready_to_update():
        log = agent.update()

    obs = next_obs  # 다음 행동은 reset된 원본 obs로

agent.save("ppo_cartpole.pt")

# 평가: 별도 eval env, 에피소드마다 고정 seed, deterministic
evaluator = Evaluator(make_eval_env(config), config)
print(evaluator.evaluate(agent))   # {'eval_return_mean': ..., 'eval_return_std': ..., 'eval_length_mean': ...}
```

Trainer가 완성되면 이 루프는 Trainer 한 줄로 대체됩니다.

## 기록 (Logger)

```python
from evalrl.core.logger import Logger

logger = Logger(config.run_dir)
logger.log("train", step, {"policy_loss": ..., "value_loss": ...})   # → run_dir/train.csv
logger.log("eval", step, evaluator.evaluate(agent))                  # → run_dir/eval.csv
logger.close()
```

이름(`"train"`, `"eval"`)마다 CSV가 따로 생기고, 첫 호출의 key가 header가 됩니다. 매 기록마다 flush하므로 학습이 중간에 끊겨도 그때까지의 기록이 남습니다.

## 테스트

```bash
python tests/test_logger_evaluator.py
```

Logger(header · 파일 분리 · flush · 새 key 오류)와 Evaluator(반환 형식 · 결정성 · 버퍼/step/가중치 무변화)를 확인합니다.

## Agent 인터페이스

모든 알고리즘은 같은 시그니처를 구현하므로, 학습 루프는 알고리즘 종류를 몰라도 됩니다.

| 메서드 | 역할 |
|---|---|
| `act(obs, deterministic) -> np.ndarray` | 배치 obs `(n_envs, obs_dim)`를 받아 행동 반환 |
| `observe(obs, next_obs, acts, rewards, terms, truncs)` | 환경 정보를 버퍼에 저장 |
| `ready_to_update() -> bool` | 업데이트 시점 판단 |
| `update() -> dict[str, float]` | 학습 1회, 지표 반환 |
| `save(path)` / `load(path)` | 네트워크 · optimizer · step 저장 및 복원 |
| `on_step(global_timestep)` | 학습 루프가 관리하는 step 수 전달 |

## 참고 자료

- Stable-Baselines3 — 전체 구조
- CleanRL — PPO 세부 구현, seed · truncation 처리
- Agarwal et al. 2021, *Deep RL at the Edge of the Statistical Precipice* (`rliable`, IQM)
- Henderson et al. 2018, *Deep RL that Matters*

---

이 README는 Anthropic의 [Claude](https://claude.ai)를 이용해 작성했습니다.
