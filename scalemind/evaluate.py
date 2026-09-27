"""Évaluation d'une politique sur les séries de test (2024, jamais vues à l'entraînement)."""

from __future__ import annotations

import numpy as np

from .env import AutoscalingEnv, EpisodeReport


def run_episode(env: AutoscalingEnv, policy, series: int, keep_trace: bool = False) -> EpisodeReport:
    obs, _ = env.reset(options={"series": series})
    policy.reset(env)
    report = EpisodeReport()
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy.act(env, obs))
        report.add(info["step"], keep_trace)
    return report


def evaluate(policy, env: AutoscalingEnv, names: list[str]) -> dict:
    """Rejoue toute l'année de test sur chaque série. Renvoie les métriques par série + le total."""
    per_series = {}
    total = EpisodeReport()
    for k, name in enumerate(names):
        rep = run_episode(env, policy, k)
        per_series[name] = rep.summary()
        total.merge(rep)
    return {"total": total.summary(), "per_series": per_series}


class ModelPolicy:
    """Adapte un modèle stable-baselines3 à l'interface des politiques."""

    def __init__(self, model, name: str):
        self.model, self.name = model, name

    def reset(self, env):
        pass

    def act(self, env, obs):
        action, _ = self.model.predict(np.asarray(obs), deterministic=True)
        return int(action)
