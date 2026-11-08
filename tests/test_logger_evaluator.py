"""
Logger · Evaluator 테스트.

실행:
    python tests/test_logger_evaluator.py      # 스크립트로
    pytest tests/test_logger_evaluator.py      # pytest로 (설치돼 있다면)

Colab 노트북에서는 이 파일 내용을 셀에 붙여넣고 맨 아래 run_all()만 호출해도 됩니다.
학습된 Agent로도 확인하려면 run_all(trained_agent=ppo) 처럼 넘기세요.
"""
import csv
import os
import tempfile
from types import SimpleNamespace

import torch

from evalrl.agents import PPOAgent
from evalrl.configs import PPOConfig
from evalrl.core.env_factory import make_eval_env
from evalrl.core.evaluator import Evaluator
from evalrl.core.logger import Logger


# ──────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_logger_basic():
    with tempfile.TemporaryDirectory() as d:
        run_dir = os.path.join(d, "runs", "test")  # 중간 폴더까지 없어도 생성되는지
        logger = Logger(run_dir)

        logger.log("train", 1000, {"loss": 0.5, "entropy": 0.69})
        logger.log("train", 2000, {"loss": 0.4, "entropy": 0.68})
        logger.log("eval", 2000, {"eval_return_mean": 120.0})

        # flush 확인: close 전에도 파일에 내용이 있어야 함
        rows = _read_csv(os.path.join(run_dir, "train.csv"))
        assert len(rows) == 2, f"close 전 train.csv 줄 수 {len(rows)} (flush 안 됨?)"

        logger.close()

        train = _read_csv(os.path.join(run_dir, "train.csv"))
        assert list(train[0].keys()) == ["step", "loss", "entropy"], f"header: {list(train[0].keys())}"
        assert [r["step"] for r in train] == ["1000", "2000"]
        assert float(train[1]["loss"]) == 0.4

        evals = _read_csv(os.path.join(run_dir, "eval.csv"))
        assert len(evals) == 1 and float(evals[0]["eval_return_mean"]) == 120.0
    print("✓ logger: header · 줄 수 · 파일 분리 · flush")


def test_logger_new_key_raises():
    with tempfile.TemporaryDirectory() as d:
        logger = Logger(d)
        logger.log("train", 1000, {"loss": 0.5})
        try:
            logger.log("train", 2000, {"loss": 0.3, "new_key": 1.0})
        except ValueError:
            print("✓ logger: 새 key는 ValueError")
        else:
            raise AssertionError("새 key가 들어왔는데 에러가 나지 않음 (조용히 누락될 수 있음)")
        finally:
            logger.close()


def test_logger_close_twice_safe():
    with tempfile.TemporaryDirectory() as d:
        logger = Logger(d)
        logger.log("train", 1, {"x": 1.0})
        logger.close()
        logger.close()  # 두 번 닫아도 에러 없어야 함
    print("✓ logger: close 두 번 호출 안전")


# ──────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────
EVAL_CFG = SimpleNamespace(env_id="CartPole-v1", eval_episodes=5, eval_seed=10_000)


def _new_agent(seed=0):
    torch.manual_seed(seed)
    env = make_eval_env(EVAL_CFG)
    agent = PPOAgent(env.single_observation_space, env.single_action_space, PPOConfig(), torch.device("cpu"))
    env.close()
    return agent


def _check_result_format(result):
    for key in ("eval_return_mean", "eval_return_std", "eval_length_mean"):
        assert key in result, f"반환 dict에 {key} 없음"
        assert type(result[key]) is float, f"{key} 타입이 {type(result[key]).__name__} (float여야 함)"


def test_evaluator_untrained(agent=None):
    agent = agent or _new_agent()
    evaluator = Evaluator(make_eval_env(EVAL_CFG), EVAL_CFG)
    result = evaluator.evaluate(agent)
    _check_result_format(result)

    assert result["eval_return_mean"] > 0, "return이 0 → 에피소드 루프가 실행되지 않음 (while 조건 확인)"
    # CartPole은 step마다 보상 1 → return == length
    assert result["eval_return_mean"] == result["eval_length_mean"], "CartPole인데 return ≠ length"
    print(f"✓ evaluator(무작위 초기화): return {result['eval_return_mean']:.1f} ± {result['eval_return_std']:.1f}")


def test_evaluator_deterministic(agent=None):
    agent = agent or _new_agent()
    evaluator = Evaluator(make_eval_env(EVAL_CFG), EVAL_CFG)
    r1 = evaluator.evaluate(agent)
    r2 = evaluator.evaluate(agent)
    assert r1 == r2, f"같은 agent · 같은 seed인데 결과가 다름\n{r1}\n{r2}"
    print("✓ evaluator: 두 번 호출 결과 동일 (seed · deterministic 정상)")


def test_evaluator_no_side_effects(agent=None):
    agent = agent or _new_agent()
    ptr_before = agent.rollout_buffer.ptr
    step_before = agent.global_timestep
    params_before = [p.detach().clone() for p in agent.clip_params]

    Evaluator(make_eval_env(EVAL_CFG), EVAL_CFG).evaluate(agent)

    assert agent.rollout_buffer.ptr == ptr_before, "평가가 버퍼에 데이터를 넣음 (observe 호출?)"
    assert agent.global_timestep == step_before, "평가가 global_timestep을 바꿈 (on_step 호출?)"
    assert all(torch.equal(a, b) for a, b in zip(params_before, agent.clip_params)), "평가가 가중치를 바꿈"
    print("✓ evaluator: 버퍼 · step · 가중치 변화 없음")


def test_evaluator_trained(trained_agent):
    result = Evaluator(make_eval_env(EVAL_CFG), EVAL_CFG).evaluate(trained_agent)
    _check_result_format(result)
    print(f"  학습된 agent: return {result['eval_return_mean']:.1f} ± {result['eval_return_std']:.1f} (CartPole이면 500 근처 기대)")


# ──────────────────────────────────────────────
def run_all(trained_agent=None):
    test_logger_basic()
    test_logger_new_key_raises()
    test_logger_close_twice_safe()
    test_evaluator_untrained()
    test_evaluator_deterministic()
    test_evaluator_no_side_effects()
    if trained_agent is not None:
        test_evaluator_trained(trained_agent)
    print("\n모든 테스트 통과")


if __name__ == "__main__":
    run_all()
