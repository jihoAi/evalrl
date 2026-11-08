from dataclasses import dataclass


@dataclass
class CommonConfig:
    # ── 공정성 항목: 비교 실험에서는 모든 알고리즘이 같은 값을 써야 함 ──
    env_id: str = "CartPole-v1"
    total_timestep: int = 500_000
    num_envs: int = 8
    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    seed: int = 0

    # ── 평가 (Evaluator) ──
    eval_episodes: int = 10
    eval_seed: int = 10_000          # 학습 seed와 겹치지 않게 큰 값
    eval_interval: int = 10_000      # 학습 env step 기준

    # ── 기록 · 저장 (Logger, Trainer) ──
    log_interval: int = 1_000
    save_interval: int = 50_000
    run_dir: str = "runs/test"


@dataclass
class A2CConfig(CommonConfig):
    num_steps: int = 5
    learning_rate: float = 7e-4
    gae_lambda: float = 1.0
    vf_coef: float = 0.5
    ent_coef: float = 0.0


@dataclass
class PPOConfig(A2CConfig):
    # A2C와 같은 필드는 기본값만 덮어씀
    num_envs: int = 4
    num_steps: int = 128
    learning_rate: float = 2.5e-4
    gae_lambda: float = 0.95
    ent_coef: float = 0.01

    # PPO 전용
    num_epochs: int = 4
    minibatch_size: int = 128
    clip_coef: float = 0.2
    norm_adv: bool = True

    def __post_init__(self):
        batch_size = self.num_envs * self.num_steps
        if batch_size % self.minibatch_size != 0:
            raise ValueError(
                f"num_envs * num_steps ({batch_size})가 "
                f"minibatch_size ({self.minibatch_size})로 나눠떨어지지 않음"
            )
