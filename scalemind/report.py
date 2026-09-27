"""Rapport final à partir des modèles du balayage (experiments/sweep.py), sans ré-entraîner.

    python -m scalemind.report

Produit : results/metrics.json, results/*.png, web/data/replay.{json,js}, models/policy.npz
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import data
from .env import AutoscalingEnv, ClusterConfig
from .evaluate import ModelPolicy, evaluate
from .policies import OraclePolicy
from .qlearning import QLearningAgent
from .train import MODELS, RESULTS, export_replay, load, tune_baselines

SWEEP = RESULTS / "sweep.jsonl"
# famille de configurations retenue pour l'agent principal (3 graines)
MAIN = "final_sh6"


def sweep_rows() -> list[dict]:
    return [json.loads(line) for line in SWEEP.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate(rows: list[dict]) -> dict:
    """Moyenne et écart-type par famille de configurations (les graines regroupées)."""
    groups = defaultdict(list)
    for r in rows:
        name = r["config"]["name"]
        fam = name.rsplit("_s", 1)[0] if name.startswith("final_") else name
        groups[fam].append(r)
    out = {}
    for fam, rs in groups.items():
        costs = np.array([r["total_cost"] for r in rs])
        out[fam] = {
            "runs": len(rs), "config": rs[0]["config"],
            "cost_mean": round(float(costs.mean()), 0), "cost_std": round(float(costs.std()), 0),
            "over_budget_mean": round(float(np.mean([r["over_budget_hours"] for r in rs])), 1),
            "healthy_mean": round(float(np.mean([r["healthy_pct"] for r in rs])), 2),
        }
    return out


def load_sb3(name: str, label: str):
    from stable_baselines3 import PPO
    return ModelPolicy(PPO.load(MODELS / "sweep" / f"{name}.zip", device="cpu"), label)


def main() -> None:
    from .serve import export_sb3

    cfg = ClusterConfig()
    train_series, test_series = load(data.PROJECTS)
    envs = {m: AutoscalingEnv(test_series, cfg, episode_hours=None, mode=m) for m in AutoscalingEnv.MODES}
    rows = sweep_rows()
    by_name = {r["config"]["name"]: r for r in rows}

    entries = [(p, "delta") for p in tune_baselines(train_series, cfg, data.PROJECTS)]

    # agent principal : la graine médiane de la famille retenue (ni la meilleure ni la pire)
    main_runs = sorted((r for r in rows if r["config"]["name"].startswith(MAIN + "_s")), key=lambda r: r["total_cost"])
    median_run = main_runs[len(main_runs) // 2]["config"]["name"]
    ppo = load_sb3(median_run, "PPO (marge)")
    entries.append((ppo, "headroom"))
    if "ablation_delta" in by_name:
        entries.append((load_sb3("ablation_delta", "PPO (±instances)"), "delta"))
    q_path = MODELS / "sweep" / "q_g99_sh3.npy"
    if q_path.exists():
        q = QLearningAgent(len(cfg.headrooms), mode="headroom")
        q.q = np.load(q_path)
        q.name = "Q-learning (marge)"
        entries.append((q, "headroom"))
    entries.append((OraclePolicy(), "delta"))

    results = {}
    for pol, mode in entries:
        res = evaluate(pol, envs[mode], data.PROJECTS)
        res["mode"] = mode
        results[pol.name] = res
        t = res["total"]
        print(f"{pol.name:28s} {t['total_cost']:10.0f} €  saines {t['healthy_pct']:6.2f} %  hors budget {t['over_budget_hours']:4d} h")

    export_sb3(ppo.model, ppo.name, MODELS / "policy.npz", "headroom")
    agg = aggregate(rows)
    meta = {"config": cfg.__dict__, "projects": data.PROJECTS, "train_period": data.TRAIN, "test_period": data.TEST,
            "main_agent": {"family": MAIN, "median_seed_run": median_run}}
    (RESULTS / "metrics.json").write_text(json.dumps({"meta": meta, "results": results, "sweep": agg}, indent=2,
                                                     ensure_ascii=False, default=list), encoding="utf-8")
    export_replay(envs, entries, data.PROJECTS, results)
    from .plots import make_plots
    make_plots(RESULTS)
    print("Rapport écrit dans results/ et web/data/")


if __name__ == "__main__":
    main()
