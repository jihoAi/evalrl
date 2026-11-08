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
| Config (dataclass) | ✅ 완료 |
| Trainer · Evaluator · Logger | 🚧 진행 중 |
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
    └── core/               (예정) env_factory, trainer, evaluator, logger
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

설치 후 **런타임 → 세션 다시 시작**을 한 번 해야 import됩니다. 코드 수정을 바로 반영하려면:

```python
%load_ext autoreload
%autoreload 2
```

## 빠른 시작

```python
import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode

from evalrl.agents import PPOAgent
from evalrl.configs import PPOConfig

config = PPOConfig(total_timestep=200_000)
device = torch.device("cpu")

envs = gym.make_vec(
    config.env_id,
    num_envs=config.num_envs,
    vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
)
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
```

Trainer가 완성되면 이 루프는 Trainer 한 줄로 대체됩니다.

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
