"""Politiques de référence (baselines) : ce que ferait un ingénieur sans RL.

Elles tournent dans un environnement en mode "delta" (ajouter/retirer des instances),
avec les mêmes coûts et les mêmes données que les agents.
"""

from __future__ import annotations

import math

import numpy as np

from .env import AutoscalingEnv


class Policy:
    name = "policy"

    def reset(self, env: AutoscalingEnv) -> None:  # noqa: B027 - hook optionnel
        pass

    def act(self, env: AutoscalingEnv, obs: np.ndarray) -> int:
        raise NotImplementedError


class StaticPolicy(Policy):
    """Provisionnement fixe pour le pic : zéro risque, beaucoup de gaspillage."""

    name = "Statique (pic)"

    def __init__(self, instances: int):
        self.instances = instances

    def act(self, env, obs):
        return env.action_for_target(self.instances)


class ReactivePolicy(Policy):
    """Autoscaler réactif type Kubernetes HPA : vise une utilisation cible sur la charge actuelle.

    Il ne regarde que le présent : sur une montée de charge, il a une heure de retard.
    """

    name = "Réactif (HPA)"

    def __init__(self, target_util: float = 0.70, scale_down_delay: int = 3):
        self.target_util = target_util
        self.scale_down_delay = scale_down_delay  # stabilisation : on attend avant de réduire
        self.history: list[int] = []

    def reset(self, env):
        self.history = []

    def act(self, env, obs):
        desired = math.ceil(env.demand[env.t] / self.target_util)
        self.history = (self.history + [desired])[-self.scale_down_delay:]
        target = desired if desired >= env.instances else max(self.history)
        return env.action_for_target(target)


class SeasonalPolicy(Policy):
    """Prévision saisonnière naïve : « même heure hier », corrigée par la tendance du jour.

    C'est une baseline forte : le trafic web est très régulier d'un jour à l'autre.
    """

    name = "Prévision saisonnière"

    def __init__(self, margin: float = 0.10):
        self.margin = margin

    def act(self, env, obs):
        return env.action_for_target(env.needed(env.forecast() * (1 + self.margin)))


class BudgetAwarePolicy(Policy):
    """Prévision saisonnière + règle SRE de « burn rate » sur le budget d'erreur.

    Marge basse tant que le budget se consomme moins vite que le temps ne passe,
    marge haute sinon. C'est la meilleure règle simple qu'un SRE écrirait.
    """

    name = "Prévision + règle SRE"

    def __init__(self, low: float = 0.0, high: float = 0.15):
        self.low, self.high = low, high

    def act(self, env, obs):
        budget_frac = env.budget_left() / env.cfg.budget_hours
        on_track = budget_frac >= 1 - env.window_progress()
        margin = self.low if on_track else self.high
        return env.action_for_target(env.needed(env.forecast() * (1 + margin)))


class OraclePolicy(Policy):
    """Triche : connaît la demande de l'heure suivante. Borne basse, pas une vraie stratégie."""

    name = "Oracle (connaît le futur)"

    def act(self, env, obs):
        return env.action_for_target(env.needed(env.demand[env.t + 1]))
