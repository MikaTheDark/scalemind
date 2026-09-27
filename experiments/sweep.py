"""Une configuration d'entraînement = un processus. Résultat en une ligne JSON.

    python experiments/sweep.py '{"name": "ppo_g999", "algo": "ppo", "gamma": 0.999, "steps": 1000000}'

Les résultats s'ajoutent à results/sweep.jsonl.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scalemind import data  # noqa: E402
from scalemind.env import AutoscalingEnv, ClusterConfig  # noqa: E402
from scalemind.evaluate import ModelPolicy, evaluate  # noqa: E402
from scalemind.qlearning import QLearningAgent  # noqa: E402
from scalemind.train import load  # noqa: E402

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent.parent


def main(cfg_json: str) -> None:
    c = json.loads(cfg_json)
    algo, steps, gamma, seed = c["algo"], c.get("steps", 1_000_000), c.get("gamma", 0.999), c.get("seed", 0)
    mode = c.get("mode", "headroom")
    base = ClusterConfig()
    train_cfg = dataclasses.replace(base, budget_shadow_price=c.get("shadow", 0.0))
    train_series, test_series = load(data.PROJECTS)
    env = AutoscalingEnv(train_series, train_cfg, episode_hours=720, seed=seed, mode=mode)
    t0 = time.time()
    if algo == "qlearning":
        agent = QLearningAgent(env.action_space.n, mode=mode, gamma=gamma, alpha=c.get("alpha", 0.05),
                               eps_decay_steps=int(steps * 0.5), seed=seed)
        agent.train(env, steps)
        policy, model = agent, agent
    else:
        from stable_baselines3 import DQN, PPO
        kw = {"net_arch": [64, 64]}
        if algo == "ppo":
            model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=c.get("n_steps", 4096), batch_size=256,
                        gamma=gamma, gae_lambda=c.get("lam", 0.98), ent_coef=c.get("ent", 0.01),
                        learning_rate=c.get("lr", 3e-4), policy_kwargs=kw)
        else:
            model = DQN("MlpPolicy", env, seed=seed, verbose=0, gamma=gamma, learning_rate=5e-4, buffer_size=200_000,
                        learning_starts=10_000, exploration_fraction=0.3, exploration_final_eps=0.02,
                        target_update_interval=4_000, train_freq=4, policy_kwargs=kw)
        model.learn(total_timesteps=steps)
        policy = ModelPolicy(model, c["name"])
    test_env = AutoscalingEnv(test_series, base, episode_hours=None, mode=mode)  # vrais coûts, sans façonnage
    res = evaluate(policy, test_env, data.PROJECTS)["total"]
    out = {"config": c, "train_s": round(time.time() - t0), **res}
    models = ROOT / "models" / "sweep"
    models.mkdir(parents=True, exist_ok=True)
    if algo == "qlearning":
        model.save(models / f"{c['name']}.npy")
    else:
        model.save(models / f"{c['name']}.zip")
    with open(ROOT / "results" / "sweep.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
