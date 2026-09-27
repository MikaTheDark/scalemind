"""Q-learning tabulaire écrit à la main (numpy), pour comprendre l'algorithme sans boîte noire.

L'état continu est discrétisé en signaux lisibles.
- Mode "headroom" (l'agent choisit une marge au-dessus de la prévision) :
  heure à venir (24) × erreur de la dernière prévision (5) × volatilité des erreurs (3)
  × avance ou retard sur le budget d'erreur (5) = 1 800 états.
- Mode "delta" (l'agent ajoute/retire des instances) :
  heure à venir (24) × marge actuelle vs besoin prévu (7) × tendance de la dernière heure (3)
  = 504 états.
"""

from __future__ import annotations

import numpy as np

from .env import AutoscalingEnv

ERR_BINS = np.array([-0.10, -0.03, 0.03, 0.10])
VOL_BINS = np.array([0.03, 0.07])
BURN_BINS = np.array([-0.3, -0.1, 0.1, 0.3])
MARGIN_BINS = np.array([0.8, 0.9, 0.97, 1.05, 1.15, 1.35])
TREND_BINS = np.array([0.97, 1.03])


def n_states(mode: str) -> int:
    if mode == "headroom":
        return 24 * (len(ERR_BINS) + 1) * (len(VOL_BINS) + 1) * (len(BURN_BINS) + 1)
    return 24 * (len(MARGIN_BINS) + 1) * (len(TREND_BINS) + 1)


def discretize(env: AutoscalingEnv) -> int:
    d, t = env.demand, env.t
    hour = int(env.series[env.k][0][t + 1].astype("datetime64[h]").astype(np.int64) % 24)
    if env.mode == "headroom":
        err = env.forecast_errors()
        e = int(np.searchsorted(ERR_BINS, err[-1]))
        v = int(np.searchsorted(VOL_BINS, np.abs(err).mean()))
        # > 0 : il reste plus de budget que de temps (on peut tourner serré)
        burn = env.budget_left() / env.cfg.budget_hours - (1 - env.window_progress())
        b = int(np.searchsorted(BURN_BINS, burn))
        return ((hour * (len(ERR_BINS) + 1) + e) * (len(VOL_BINS) + 1) + v) * (len(BURN_BINS) + 1) + b
    margin = env.instances / max(1.0, env.needed(env.forecast()))
    trend = d[t] / max(1e-9, d[t - 1])
    m = int(np.searchsorted(MARGIN_BINS, margin))
    tr = int(np.searchsorted(TREND_BINS, trend))
    return (hour * (len(MARGIN_BINS) + 1) + m) * (len(TREND_BINS) + 1) + tr


class QLearningAgent:
    name = "Q-learning (tabulaire)"

    def __init__(self, n_actions: int, mode: str = "headroom", alpha: float = 0.05, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.02, eps_decay_steps: int = 150_000, seed: int = 0):
        self.mode = mode
        self.q = np.zeros((n_states(mode), n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def choose(self, s: int, explore: bool) -> int:
        # epsilon-greedy : un peu de hasard pour explorer, sinon la meilleure action connue
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.q.shape[1]))
        return int(np.argmax(self.q[s]))

    def train(self, env: AutoscalingEnv, total_steps: int, log_every: int = 10_000) -> list[tuple[int, float]]:
        curve, returns, ep_return = [], [], 0.0
        env.reset()
        s = discretize(env)
        while self.steps < total_steps:
            a = self.choose(s, explore=True)
            _, r, done, _, _ = env.step(a)
            ep_return += r
            if done:
                target = r
                returns.append(ep_return)
                ep_return = 0.0
                env.reset()
                s2 = discretize(env)
            else:
                s2 = discretize(env)
                target = r + self.gamma * self.q[s2].max()
            self.q[s, a] += self.alpha * (target - self.q[s, a])
            s = s2
            self.steps += 1
            if self.steps % log_every == 0 and returns:
                curve.append((self.steps, float(np.mean(returns[-20:]))))
        return curve

    # interface commune avec les autres politiques
    def reset(self, env):
        pass

    def act(self, env, obs):
        return self.choose(discretize(env), explore=False)

    def save(self, path) -> None:
        np.save(path, self.q)
