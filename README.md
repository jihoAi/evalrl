# evalrl

여러 강화학습 에이전트의 학습 과정과 결과를 **일관된 지표와 형식으로** 기록·비교하기 위한 PyTorch 라이브러리입니다.

직접 구현한 DQN · A2C · PPO를 하나의 공통 프레임워크에서 학습·평가합니다. 알고리즘마다 학습 방식(off-policy / on-policy, 버퍼, 업데이트 주기)이 달라 조건을 완전히 같게 만들 수는 없으므로, evalrl은 알고리즘 간 우열을 판정하는 벤치마크가 아닙니다. 대신 **무엇을 어떻게 측정하고 기록할지를 통일**해서, 서로 다른 알고리즘의 결과를 같은 틀 위에 놓고 볼 수 있게 하는 것이 목표입니다.

## 비교 기준

### 통일하는 것 (측정 · 기록 방식)

| 항목 | 통일 방법 |
|---|---|
| x축 | 모든 지표를 환경 step 수 기준으로 기록 |
| 학습 예산 | 같은 환경 step 수 |
| 시드 | 같은 seed 집합, 알고리즘당 5~10개 |
| 환경 | 같은 `env_id` · `num_envs`, SAME_STEP autoreset (`configs/{env}/common.yaml`) |
| 종료 처리 | truncation에서는 bootstrap, termination에서는 안 함 (세 알고리즘 동일) |
| 평가 프로토콜 | 별도 eval env, 고정 eval seed, deterministic policy, 같은 평가 주기 · 에피소드 수 |
| 네트워크 생성 | 공통 팩토리 `build_mlp`, 같은 hidden 구조 |
| 출력 형식 | 같은 파일 구성(`config.yaml`, `train.csv`, `eval.csv`)과 같은 지표 이름 |

### 통일하지 않는 것 (알고리즘 고유)

- 알고리즘별 하이퍼파라미터 (learning rate, rollout 길이, replay buffer 크기, target update 주기 등)
- 데이터 수집 · 업데이트 방식 (off-policy replay vs on-policy rollout)
- 네트워크 출력 구조 (Q-network vs actor-critic)

이 차이는 숨기지 않고 각 실행의 `config.yaml`에 그대로 기록합니다. 따라서 결과는 "어느 알고리즘이 더 좋다"는 판정이 아니라, **이 설정에서 각 알고리즘이 이렇게 학습했다**는 일관된 기록으로 읽어야 합니다.

환경은 DQN이 이산 행동(discrete action)만 지원하므로 CartPole · LunarLander를 사용합니다.

## 디렉토리 구조

```
evalrl/                     (저장소 루트)
├── pyproject.toml
├── README.md
├── configs/                실험 설정 YAML (환경별 폴더)
│   ├── cartpole/
│   │   ├── common.yaml     공통 비교 기준 (예산, 환경, 네트워크, 평가)
│   │   ├── dqn.yaml
│   │   ├── a2c.yaml
│   │   └── ppo.yaml
│   └── lunarlander/        (같은 구성)
├── tests/
│   └── test_logger_evaluator.py
└── evalrl/                 (패키지)
    ├── __init__.py
    ├── configs.py          CommonConfig, DQNConfig, A2CConfig, PPOConfig (dataclass)
    ├── config_loader.py    load_config, save_config
    ├── runner.py           run_one, run_all (algo × seed 실행)
    ├── analyze.py          analyze (IQM · 95% CI 요약, 학습 곡선)
    ├── agents/
    │   ├── __init__.py
    │   ├── base.py         Agent (추상 클래스)
    │   ├── dqn.py          DQNAgent
    │   ├── a2c.py          A2CAgent
    │   └── ppo.py          PPOAgent (A2CAgent 상속)
    ├── utils/
    │   ├── __init__.py
    │   ├── build_mlp.py
    │   ├── replay_buffer.py
    │   ├── rollout_buffer.py
    │   └── compute_advantages.py
    └── core/
        ├── __init__.py
        ├── env_factory.py  make_train_envs, make_eval_env
        ├── evaluator.py    Evaluator
        ├── logger.py       Logger
        └── trainer.py      Trainer
```

## 설치

```bash
git clone https://github.com/jihoAi/evalrl.git
pip install -e evalrl
```

`-e`(editable) 설치라 코드를 고치면 재설치 없이 반영됩니다.

추가로 필요한 패키지:

```bash
pip install matplotlib            # analyze (학습 곡선 그래프)
pip install "gymnasium[box2d]"     # LunarLander
```

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

코드를 고친 뒤에는 GitHub에 commit → 위 셀(pull) 실행 → 세션 다시 시작 순서로 반영합니다. 세션을 다시 시작하면 import도 다시 실행해야 합니다. Colab(Python 3.12+)에서는 `%autoreload`가 `imp` 모듈 오류로 동작하지 않으므로 쓰지 않습니다. `!python ...`으로 실행하는 스크립트는 매번 새 프로세스라 재시작 없이 최신 코드가 반영됩니다.

> Colab에서 파일을 직접 고치면 다음 `git pull`이 거부될 수 있습니다. 수정은 GitHub 쪽에서만 하고 Colab은 pull만 하세요.

## 빠른 시작

### 여러 알고리즘 × seed 실행 (Runner)

```python
from evalrl.runner import run_all

results = run_all(
    algos=["dqn", "a2c", "ppo"],
    seeds=[0, 1, 2, 3, 4],
    config_dir="/content/repos/evalrl/configs/cartpole",   # 로컬이면 "configs/cartpole"
    runs_root="runs/cartpole",
)
```

- 실행마다 `runs_root/{algo}_seed{n}/` 폴더가 생깁니다.
- `final.pt`가 있는 run은 건너뛰고, 중간에 끊긴 run은 지우고 다시 학습합니다. Colab 세션이 끊겨도 같은 셀을 다시 실행하면 이어서 진행됩니다.
- 한 run만 돌릴 때는 `run_one(algo, seed, ...)`을 씁니다.

### 결과 비교 (analyze)

```python
from evalrl.analyze import analyze

summary = analyze("runs/cartpole")
```

끝난 run(`final.pt`가 있는 폴더)의 `eval.csv`를 모아 seed 축으로 **IQM과 95% bootstrap CI**를 계산합니다. 결과는 `runs_root/analysis/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `learning_curve.png` | 학습 곡선 (x축: 환경 step, y축: IQM eval return ± CI) |
| `summary.md` | 요약 지표 표 |
| `chart.json` | 블로그 차트용 데이터 |

요약 지표:

| 지표 | 의미 |
|---|---|
| Normalized AUC | 학습 곡선 아래 면적 ÷ step 구간 = 학습 중 평균 성능 (표본 효율, sample efficiency) |
| Final return | 마지막 3개 평가의 평균 (최종 성능) |
| Steps to threshold | 해결 기준(CartPole 475, LunarLander 200)에 처음 도달한 step. 도달 못 한 seed는 마지막 step으로 처리 |

> 알고리즘마다 seed별 평가 시점이 같아야 묶을 수 있으므로, `total_timestep` · `eval_interval`은 `common.yaml`에서만 바꿉니다.

### 단일 실행 (구성 요소 직접 조립)

```python
import torch

from evalrl.agents import PPOAgent
from evalrl.config_loader import load_config
from evalrl.core.env_factory import make_train_envs, make_eval_env
from evalrl.core.evaluator import Evaluator
from evalrl.core.logger import Logger
from evalrl.core.trainer import Trainer

config = load_config(
    "ppo",
    overrides={"seed": 0, "run_dir": "runs/ppo_seed0"},
    config_dir="/content/repos/evalrl/configs/cartpole",   # 로컬이면 "configs/cartpole"
)

envs = make_train_envs(config)
agent = PPOAgent(envs.single_observation_space, envs.single_action_space, config, torch.device("cpu"))
evaluator = Evaluator(make_eval_env(config), config)
logger = Logger(config.run_dir)

result = Trainer(agent, envs, config, evaluator, logger).train()
print(result)   # {'eval_return_mean': ..., 'eval_return_std': ..., 'eval_length_mean': ...}
```

학습이 끝나면 `run_dir`에 다음 파일이 남습니다.

| 파일 | 내용 |
|---|---|
| `config.yaml` | 이 실행에 쓰인 최종 설정 (재현용) |
| `train.csv` | 학습 지표 (return, loss, entropy, sps 등) |
| `eval.csv` | 평가 지표 (`eval_return_mean` 등) |
| `latest.pt` | 주기적 체크포인트 |
| `final.pt` | 학습 종료 시점 모델 |

## 설정 (Config)

설정은 아래 순서로 덮어써집니다. 뒤로 갈수록 우선순위가 높습니다.

```
dataclass 기본값 (configs.py) < configs/{env}/common.yaml < configs/{env}/{algo}.yaml < overrides
```

- `common.yaml`: 모든 알고리즘에 공통으로 적용하는 비교 기준 (환경, 예산, `num_envs`, 네트워크, 평가)
- `{algo}.yaml`: 알고리즘 고유 하이퍼파라미터만
- 환경은 `config_dir`로 고릅니다 (`configs/cartpole`, `configs/lunarlander`). 기본값은 `configs/cartpole`.
- `overrides`: 실행마다 달라지는 값 (`seed`, `run_dir`) 또는 빠른 테스트용 값 (`total_timestep`)

YAML에 없는 key를 넣으면 dataclass 생성 시 `TypeError`가 나서 오타가 바로 드러납니다.

> YAML에서 `1e-3` 같은 지수 표기는 문자열로 읽히므로 `0.001` 또는 `1.0e-3`으로 씁니다.

## Agent 인터페이스

모든 알고리즘은 같은 시그니처를 구현하므로, Trainer는 알고리즘 종류를 몰라도 됩니다.

| 메서드 | 역할 |
|---|---|
| `act(obs, deterministic) -> np.ndarray` | 배치 obs `(n_envs, obs_dim)`를 받아 행동 반환 |
| `observe(obs, next_obs, acts, rewards, terms, truncs)` | 환경 정보를 버퍼에 저장 |
| `ready_to_update() -> bool` | 업데이트 시점 판단 |
| `update() -> dict[str, float]` | 학습 1회, 지표 반환 |
| `save(path)` / `load(path)` | 네트워크 · optimizer · step 저장 및 복원 |
| `on_step(global_timestep)` | 학습 루프가 관리하는 step 수 전달 |

## 학습 흐름 (Trainer)

```
act → env.step → (SAME_STEP final_obs 교체) → observe → on_step → ready_to_update → update
                                                    ↓ 주기마다
                              log (train.csv) · eval (eval.csv) · save (latest.pt)
```

- step은 환경 step 기준 (`num_envs`씩 증가), 비교 그래프의 x축
- 학습 전 · 주기적 · 종료 시 평가 (같은 step에서 중복 없이)
- 에러나 중단이 나도 `finally`에서 Logger를 닫아 기록이 보존됨

## 테스트

```bash
python tests/test_logger_evaluator.py
```

Logger(header · 파일 분리 · flush · 새 key 오류)와 Evaluator(반환 형식 · 결정성 · 버퍼/step/가중치 무변화)를 확인합니다.

## 참고 자료

- Stable-Baselines3 — 전체 구조
- CleanRL — PPO 세부 구현, seed · truncation 처리
- Agarwal et al. 2021, *Deep RL at the Edge of the Statistical Precipice* (`rliable`, IQM)
- Henderson et al. 2018, *Deep RL that Matters*

---

이 README는 Anthropic의 [Claude](https://claude.ai)를 이용해 작성했습니다.
