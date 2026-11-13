"""
Runner가 남긴 결과(runs_root/{algo}_seed{n}/eval.csv)를 모아 알고리즘을 비교한다.

    from evalrl.analyze import analyze
    summary = analyze("/content/drive/MyDrive/evalrl/runs/cartpole")

out_dir(기본: runs_root/analysis)에 세 파일을 만든다.
    learning_curve.png  학습 곡선 (IQM + 95% bootstrap CI)
    chart.json          블로그 차트용 데이터 (assets/data/에 그대로 올리는 형식)
    summary.md          요약 지표 표 (블로그 글에 붙여넣는 마크다운)
"""
import csv
import json
import os
import re

import numpy as np
import yaml

# 환경별 기본값: 해결 기준(threshold)과 그래프 세로축 범위
ENV_DEFAULTS = {
    "CartPole-v1": {"threshold": 475, "ymin": 0, "ymax": 500},
    "LunarLander-v3": {"threshold": 200, "ymin": None, "ymax": None},
}

RUN_DIR_PATTERN = re.compile(r"^(?P<algo>.+)_seed(?P<seed>\d+)$")


# ──────────────────────────────────────────────
# 통계
# ──────────────────────────────────────────────
def iqm(x, axis=0):
    """사분위 평균 (interquartile mean): 아래 · 위 25%를 잘라내고 평균. scipy.stats.trim_mean(x, 0.25)와 같다."""
    x = np.sort(np.asarray(x, dtype=float), axis=axis)
    n = x.shape[axis]
    cut = int(0.25 * n)
    return np.take(x, np.arange(cut, n - cut), axis=axis).mean(axis=axis)


def iqm_ci(x, reps=10_000, alpha=0.05, seed=0):
    """x: (num_seeds, ...) -> (IQM, CI 하한, CI 상한). seed 축으로 bootstrap."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(reps, n))
    boot = iqm(x[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return iqm(x, axis=0), lo, hi


def _trapezoid(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.0에서 이름이 바뀜
    return fn(y, x, axis=1)


# ──────────────────────────────────────────────
# 불러오기
# ──────────────────────────────────────────────
def _read_eval(run_dir):
    with open(os.path.join(run_dir, "eval.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(float(r["step"])) for r in rows])
    returns = np.array([float(r["eval_return_mean"]) for r in rows])
    return steps, returns


def load_runs(runs_root, algos=None, seeds=None):
    """
    끝난 run(final.pt가 있는 폴더)만 읽는다.
    반환: {algo: {"steps": (n_points,), "returns": (n_seeds, n_points), "seeds": [..]}}
    algos · seeds를 안 주면 폴더 이름에서 찾는다. algos를 주면 그 순서가 그래프 순서가 된다.
    """
    found = {}
    for name in sorted(os.listdir(runs_root)):
        match = RUN_DIR_PATTERN.match(name)
        run_dir = os.path.join(runs_root, name)
        if not match or not os.path.isdir(run_dir):
            continue
        algo, seed = match["algo"], int(match["seed"])
        if algos is not None and algo not in algos:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if not os.path.exists(os.path.join(run_dir, "final.pt")):
            print(f"건너뜀 (아직 안 끝남): {name}")
            continue
        found.setdefault(algo, {})[seed] = _read_eval(run_dir)

    if not found:
        raise FileNotFoundError(f"{runs_root}에 끝난 run이 없음 (final.pt가 있는 {{algo}}_seed{{n}} 폴더)")

    data = {}
    for algo in (algos if algos is not None else sorted(found)):
        if algo not in found:
            print(f"건너뜀 (끝난 run 없음): {algo}")
            continue
        runs = found[algo]
        seed_list = sorted(runs)
        steps = runs[seed_list[0]][0]
        for s in seed_list:
            if not np.array_equal(runs[s][0], steps):
                raise ValueError(f"{algo}: seed {s}의 평가 시점이 seed {seed_list[0]}과 다름 (예산 · eval_interval이 같은지 확인)")
        data[algo] = {"steps": steps, "returns": np.stack([runs[s][1] for s in seed_list]), "seeds": seed_list}
    return data


def _env_id(runs_root):
    for name in sorted(os.listdir(runs_root)):
        path = os.path.join(runs_root, name, "config.yaml")
        if RUN_DIR_PATTERN.match(name) and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f).get("env_id")
    return None


# ──────────────────────────────────────────────
# 요약 지표
# ──────────────────────────────────────────────
def summarize(steps, returns, threshold=None, last_k=3):
    """
    seed별 지표를 구한 뒤 IQM과 95% CI로 묶는다. 반환값의 각 지표는 (IQM, 하한, 상한).
        auc                 정규화 AUC = 학습 중 평균 성능 (표본 효율)
        final               마지막 last_k개 평가의 평균 (최종 성능)
        steps_to_threshold  threshold에 처음 도달한 step. 도달 못 한 seed는 마지막 step으로 처리
    """
    auc = _trapezoid(returns, steps) / (steps[-1] - steps[0])
    final = returns[:, -last_k:].mean(axis=1)
    out = {
        "n_seeds": int(returns.shape[0]),
        "auc": tuple(float(v) for v in iqm_ci(auc)),
        "final": tuple(float(v) for v in iqm_ci(final)),
    }
    if threshold is not None:
        solved = returns >= threshold
        reached = solved.any(axis=1)
        to_threshold = np.where(reached, steps[solved.argmax(axis=1)], steps[-1])
        out["steps_to_threshold"] = tuple(float(v) for v in iqm_ci(to_threshold))
        out["n_reached"] = int(reached.sum())
    return out


def summary_markdown(summary, threshold=None):
    def cell(v, digits):
        return f"{v[0]:,.{digits}f} [{v[1]:,.{digits}f}, {v[2]:,.{digits}f}]"

    header = ["Algorithm", "Seeds", "Normalized AUC", "Final return"]
    if threshold is not None:
        header += [f"Steps to {threshold:g}", "Reached"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for algo, s in summary.items():
        row = [algo.upper(), str(s["n_seeds"]), cell(s["auc"], 1), cell(s["final"], 1)]
        if threshold is not None:
            row += [cell(s["steps_to_threshold"], 0), f"{s['n_reached']}/{s['n_seeds']}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n\nIQM [95% bootstrap CI]\n"


# ──────────────────────────────────────────────
# 블로그 차트 데이터
# ──────────────────────────────────────────────
def chart_data(data, title=None, threshold=None, ymin=None, ymax=None):
    """블로그(jihoAi.github.io)의 차트 형식 dict. 선은 5개까지 색이 구분된다."""
    chart = {"title": title, "xlabel": "Environment steps", "ylabel": "IQM evaluation return (95% CI)",
             "threshold": threshold, "ymin": ymin, "ymax": ymax}
    chart = {k: v for k, v in chart.items() if v is not None}
    chart["series"] = []
    for algo, d in data.items():
        m, lo, hi = iqm_ci(d["returns"])
        chart["series"].append({
            "name": f"{algo.upper()} (n={len(d['seeds'])})",
            "x": [int(v) for v in d["steps"]],
            "y": np.round(m, 2).tolist(),
            "lower": np.round(lo, 2).tolist(),
            "upper": np.round(hi, 2).tolist(),
        })
    return chart


def chart_json(chart):
    """붙여넣기 좋게 series를 한 줄에 하나씩 쓴 JSON 문자열."""
    head = {k: v for k, v in chart.items() if k != "series"}
    lines = [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
    series = ",\n".join("    " + json.dumps(s, ensure_ascii=False) for s in chart["series"])
    return "{\n" + "\n".join(lines) + '\n  "series": [\n' + series + "\n  ]\n}\n"


# ──────────────────────────────────────────────
# 그래프
# ──────────────────────────────────────────────
def plot_learning_curves(data, title=None, threshold=None, ymin=None, ymax=None, path=None, show=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, d in data.items():
        m, lo, hi = iqm_ci(d["returns"])
        ax.plot(d["steps"], m, lw=2, label=f"{algo.upper()} (n={len(d['seeds'])})")
        ax.fill_between(d["steps"], lo, hi, alpha=0.2)
    if threshold is not None:
        ax.axhline(threshold, ls="--", c="k", lw=0.8)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("IQM evaluation return (95% CI)")
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, None if ymax is None else ymax * 1.04)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 한 번에 실행
# ──────────────────────────────────────────────
def analyze(runs_root, algos=None, seeds=None, title=None, threshold=None, ymin=None, ymax=None,
            out_dir=None, last_k=3, show=True):
    """
    불러오기 → 요약 지표 → 그래프 · 차트 JSON · 표 저장. 반환: {algo: summarize() 결과}
    title · threshold · ymin · ymax를 안 주면 config.yaml의 env_id에 맞는 기본값(ENV_DEFAULTS)을 쓴다.
    """
    data = load_runs(runs_root, algos=algos, seeds=seeds)

    env_id = _env_id(runs_root)
    defaults = ENV_DEFAULTS.get(env_id, {})
    title = title if title is not None else env_id
    threshold = threshold if threshold is not None else defaults.get("threshold")
    ymin = ymin if ymin is not None else defaults.get("ymin")
    ymax = ymax if ymax is not None else defaults.get("ymax")

    n_seeds = {algo: len(d["seeds"]) for algo, d in data.items()}
    if len(set(n_seeds.values())) > 1:
        print(f"주의: 알고리즘마다 끝난 seed 수가 다름 {n_seeds}")

    summary = {algo: summarize(d["steps"], d["returns"], threshold, last_k) for algo, d in data.items()}
    table = summary_markdown(summary, threshold)
    chart = chart_json(chart_data(data, title, threshold, ymin, ymax))

    out_dir = out_dir or os.path.join(runs_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table)
    with open(os.path.join(out_dir, "chart.json"), "w", encoding="utf-8") as f:
        f.write(chart)
    plot_learning_curves(data, title, threshold, ymin, ymax, os.path.join(out_dir, "learning_curve.png"), show)

    print(table)
    print(f"저장: {out_dir} (learning_curve.png, chart.json, summary.md)")
    return summary
