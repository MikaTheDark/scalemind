"""Entraîne les agents sur 2023, les évalue sur 2024 et compare aux baselines.

    python -m scalemind.train --algos qlearning ppo dqn --steps 300000

Sorties : models/, results/metrics.json, results/*.png, web/data/replay.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import data
from .env import AutoscalingEnv, ClusterConfig
from .evaluate import ModelPolicy, evaluate, run_episode
from .policies import BudgetAwarePolicy, OraclePolicy, ReactivePolicy, SeasonalPolicy, StaticPolicy
from .qlearning import QLearningAgent

ROOT = Path(__file__).resolve().parent.parent
MODELS, RESULTS, WEB_DATA = ROOT / "models", ROOT / "results", ROOT / "web" / "data"


def load(projects: list[str]):
    train, test = [], []
    for p in projects:
        tr, te = data.split(p)
        # les 168 premières heures de test servent d'historique : on les prend en fin de 2023
        te_full = np.concatenate([tr.values[-AutoscalingEnv.LAGS:], te.values])
        ts_full = np.concatenate([tr.index.values[-AutoscalingEnv.LAGS:], te.index.values])
        train.append((tr.index.values, tr.values))
        test.append((ts_full, te_full))
    return train, test


def tune_baselines(train_series, cfg: ClusterConfig, names: list[str]) -> list:
    """Règle chaque baseline sur 2023 (grille), pour affronter leur meilleure version."""
    env = AutoscalingEnv(train_series, cfg, episode_hours=None, mode="delta")

    def cost(p):
        return evaluate(p, env, names)["total"]["total_cost"]

    peak = max(float(np.quantile(d, 0.99)) for _, d in train_series)
    static = min((StaticPolicy(math.ceil(peak * f / cfg.target_util)) for f in (0.9, 1.0, 1.1, 1.2)), key=cost)
    reactive = min((ReactivePolicy(u) for u in (0.5, 0.6, 0.7, 0.8)), key=cost)
    seasonal = min((SeasonalPolicy(m) for m in (0.0, 0.03, 0.05, 0.08, 0.12, 0.2)), key=cost)
    sre = min((BudgetAwarePolicy(lo, hi) for lo in (-0.05, 0.0, 0.03, 0.06) for hi in (0.08, 0.12, 0.2, 0.3)), key=cost)
    print(f"Baselines réglées sur 2023 : statique={static.instances} inst., HPA cible={reactive.target_util}, "
          f"marge saisonnière={seasonal.margin}, règle SRE={sre.low}/{sre.high}")
    return [static, reactive, seasonal, sre]


def train_sb3(algo: str, env: AutoscalingEnv, steps: int, seed: int):
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor

    menv = Monitor(env)
    if algo == "ppo":
        model = PPO("MlpPolicy", menv, seed=seed, verbose=0, n_steps=2048, batch_size=256,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4, policy_kwargs={"net_arch": [64, 64]})
    else:
        model = DQN("MlpPolicy", menv, seed=seed, verbose=0, gamma=0.99, learning_rate=5e-4,
                    buffer_size=100_000, learning_starts=5_000, exploration_fraction=0.3,
                    exploration_final_eps=0.02, target_update_interval=2_000, train_freq=4,
                    policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=steps)
    # courbe d'apprentissage : retour moyen par épisode (fenêtre glissante)
    rets = np.array(menv.get_episode_rewards())
    lens = np.cumsum(menv.get_episode_lengths())
    win = 20
    curve = [(int(lens[i]), float(rets[max(0, i - win + 1): i + 1].mean())) for i in range(0, len(rets), max(1, len(rets) // 100))]
    return model, curve


AGENT_NAMES = {
    "qlearning": "Q-learning (marge)",
    "dqn": "DQN (marge)",
    "ppo": "PPO (marge)",
    "ppo-delta": "PPO (±instances)",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--projects", nargs="*", default=data.PROJECTS)
    p.add_argument("--algos", nargs="*", default=["qlearning", "dqn", "ppo", "ppo-delta"], choices=list(AGENT_NAMES))
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = ClusterConfig()
    train_series, test_series = load(args.projects)
    test_envs = {m: AutoscalingEnv(test_series, cfg, episode_hours=None, mode=m) for m in AutoscalingEnv.MODES}
    for d in (MODELS, RESULTS, WEB_DATA):
        d.mkdir(parents=True, exist_ok=True)

    # (politique, mode d'environnement) : les baselines agissent en « delta »
    entries = [(pol, "delta") for pol in tune_baselines(train_series, cfg, args.projects)]
    curves = {}
    for algo in args.algos:
        t0 = time.time()
        mode = "delta" if algo.endswith("-delta") else "headroom"
        train_env = AutoscalingEnv(train_series, cfg, episode_hours=720, seed=args.seed, mode=mode)
        name = AGENT_NAMES[algo]
        if algo == "qlearning":
            agent = QLearningAgent(train_env.action_space.n, mode=mode, seed=args.seed)
            agent.name = name
            curves[name] = agent.train(train_env, args.steps)
            agent.save(MODELS / "qlearning.npy")
            entries.append((agent, mode))
        else:
            model, curve = train_sb3(algo.split("-")[0], train_env, args.steps, args.seed)
            model.save(MODELS / f"{algo}.zip")
            curves[name] = curve
            entries.append((ModelPolicy(model, name), mode))
        print(f"[{algo}] entraîné en {time.time() - t0:.0f}s")
    entries.append((OraclePolicy(), "delta"))

    results = {}
    print(f"\n{'Politique':28s} {'Coût total €':>12s} {'Infra €':>10s} {'Pénalités €':>12s} {'Saines %':>9s} {'H hors budget':>13s} {'Util %':>7s}")
    for pol, mode in entries:
        res = evaluate(pol, test_envs[mode], args.projects)
        res["mode"] = mode
        results[pol.name] = res
        t = res["total"]
        print(f"{pol.name:28s} {t['total_cost']:12.0f} {t['infra_cost']:10.0f} {t['sla_cost']:12.0f} {t['healthy_pct']:9.2f} {t['over_budget_hours']:13d} {t['avg_util_pct']:7.1f}")

    # le meilleur agent réseau de neurones est exporté pour l'API (inférence numpy, sans PyTorch)
    nn_entries = [(pol, mode) for pol, mode in entries if isinstance(pol, ModelPolicy)]
    if nn_entries:
        from .serve import export_sb3
        pol, mode = min(nn_entries, key=lambda e: results[e[0].name]["total"]["total_cost"])
        export_sb3(pol.model, pol.name, MODELS / "policy.npz", mode)
        print(f"Politique exportée pour l'API : {pol.name} -> models/policy.npz")

    meta = {"config": cfg.__dict__, "projects": args.projects, "steps": args.steps, "seed": args.seed,
            "train_period": data.TRAIN, "test_period": data.TEST}
    (RESULTS / "metrics.json").write_text(json.dumps({"meta": meta, "results": results, "curves": curves}, indent=2, ensure_ascii=False, default=list), encoding="utf-8")
    export_replay(test_envs, entries, args.projects, results)
    try:
        from .plots import make_plots
        make_plots(RESULTS)
    except ImportError:
        pass


def export_replay(envs: dict, entries, names: list[str], results: dict, weeks: int = 8) -> None:
    """Exporte quelques semaines de 2024 heure par heure, pour la visualisation web."""
    hours = weeks * 168
    cfg = next(iter(envs.values())).cfg
    out = {"hours": hours, "config": cfg.__dict__, "series": {}, "summary": {}}
    for k, name in enumerate(names):
        traces, ts, demand = {}, None, None
        for pol, mode in entries:
            env = envs[mode]
            rep = run_episode(env, pol, k, keep_trace=True)
            tr = rep.trace[:hours]
            traces[pol.name] = [n for _, n in tr]
            if ts is None:
                start = AutoscalingEnv.LAGS + 1
                ts = [str(x)[:13] for x in env.series[k][0][start: start + hours]]
                demand = [round(dd, 3) for dd, _ in tr]
        out["series"][name] = {"timestamps": ts, "demand": demand, "instances": traces}
    out["summary"] = {pol: res["total"] for pol, res in results.items()}
    payload = json.dumps(out, ensure_ascii=False, default=list)
    (WEB_DATA / "replay.json").write_text(payload, encoding="utf-8")
    # variante <script> : la page fonctionne aussi ouverte directement depuis le disque (file://)
    (WEB_DATA / "replay.js").write_text(f"window.REPLAY = {payload};\n", encoding="utf-8")


if __name__ == "__main__":
    main()
