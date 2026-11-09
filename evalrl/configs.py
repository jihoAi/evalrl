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

    def __post_init__(self):
        self.hidden = tuple(self.hidden)

@dataclass
class DQNConfig(CommonConfig):
    max_grad_norm: float = 10.0     

    # DQN 전용 (모든 step 값은 env step 기준)
    buffer_size: int = 10_000
    batch_size: int = 64
    learning_rate: float = 5e-4
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_steps: int = 10_000
    learning_starts: int = 10_000
    train_freq: int = 10
    target_update_freq: int = 1_000

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size})가 buffer_size ({self.buffer_size})보다 큼"
            )
        if self.learning_starts < self.batch_size:
            raise ValueError(
                f"learning_starts ({self.learning_starts})가 batch_size ({self.batch_size})보다 작음"
            )
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ValueError(
                f"0 <= eps_end ({self.eps_end}) <= eps_start ({self.eps_start}) <= 1 이어야 함"
            )


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
        super().__post_init__()
        batch_size = self.num_envs * self.num_steps
        if batch_size % self.minibatch_size != 0:
            raise ValueError(
                f"num_envs * num_steps ({batch_size})가 "
                f"minibatch_size ({self.minibatch_size})로 나눠떨어지지 않음"
            )
