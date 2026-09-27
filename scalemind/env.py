"""Environnement Gymnasium : autoscaling d'un cluster cloud sous budget d'erreur (SLO), sur trafic réel.

Chaque pas = une heure. À la fin de l'heure t, l'autoscaler connaît la demande d_t et
choisit la capacité de l'heure t+1 ; les nouvelles instances démarrent pendant ce temps
(démarrage facturé). Il faut donc anticiper.

Le contrat de service (SLO) impose qu'au moins `slo_target` des heures de chaque fenêtre
de 30 jours soient « saines » (utilisation ≤ `target_util`). Les heures dégradées
autorisées forment le **budget d'erreur** :
  - tant qu'il reste du budget, une heure dégradée ne coûte rien de plus ;
  - une fois le budget épuisé, chaque heure dégradée coûte `over_budget_penalty` €
    (avoirs contractuels, image de marque).
Les requêtes réellement perdues (demande > 100 % de la capacité) coûtent toujours `drop_cost`.

La bonne stratégie dépend donc de l'état du budget : tourner serré quand il en reste,
se protéger quand il s'épuise. C'est une décision séquentielle à effet différé, le terrain
naturel du reinforcement learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ClusterConfig:
    min_instances: int = 1
    max_instances: int = 40
    price_per_hour: float = 0.20  # € par instance-heure (ordre de grandeur d'une VM 4 vCPU)
    boot_cost: float = 0.05  # € par instance démarrée (minutes facturées pendant le boot)
    target_util: float = 0.85  # au-delà, la latence se dégrade : heure « dégradée »
    slo_target: float = 0.97  # part minimale d'heures saines par fenêtre
    window_hours: int = 720  # fenêtre de SLO : 30 jours
    over_budget_penalty: float = 15.0  # € par heure dégradée une fois le budget épuisé
    drop_cost: float = 1.0  # € par instance-heure de demande non servie (> 100 % de capacité)
    # mode "delta" : l'agent ajoute ou retire des instances
    actions: tuple[int, ...] = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
    # mode "headroom" : l'agent choisit la marge de sécurité au-dessus de la prévision
    headrooms: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40)
    reward_scale: float = 0.2
    # Façonnage de la récompense (entraînement seulement) : « prix fictif » d'une heure dégradée
    # consommée dans le budget. Rend immédiat un coût qui, sinon, n'apparaît que des semaines plus tard.
    # L'évaluation utilise toujours les vrais coûts.
    budget_shadow_price: float = 0.0

    @property
    def budget_hours(self) -> float:
        return (1 - self.slo_target) * self.window_hours


@dataclass
class StepInfo:
    demand: float
    instances: int
    infra_cost: float
    sla_cost: float  # pénalités hors budget + requêtes perdues
    degraded: bool
    over_budget: bool
    unserved: float  # demande au-delà de 100 % de la capacité


def hour_features(ts: np.ndarray) -> np.ndarray:
    """Encodage cyclique de l'heure et du jour de la semaine (sin, cos)."""
    hours = ts.astype("datetime64[h]").astype(np.int64)
    hod = hours % 24
    dow = (hours // 24 + 3) % 7  # 1970-01-01 était un jeudi
    return np.stack(
        [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
         np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)],
        axis=1,
    ).astype(np.float32)


class AutoscalingEnv(gym.Env):
    """Autoscaling horaire sur une ou plusieurs séries de demande (en instances nécessaires).

    `series` : liste de tuples (timestamps numpy datetime64[h], demande float).
    En entraînement, chaque épisode tire une série et une fenêtre SLO complète au hasard.
    En évaluation (`episode_hours=None`), l'épisode parcourt toute la série ; le budget
    d'erreur est remis à zéro toutes les `window_hours`.
    """

    metadata = {"render_modes": []}
    LAGS = 192  # 8 jours d'historique : une semaine de décalage + 24 h d'erreurs de prévision
    OBS_DIM = 16
    MODES = ("delta", "headroom")

    def __init__(self, series, config: ClusterConfig | None = None, episode_hours: int | None = 720,
                 seed: int | None = None, mode: str = "headroom"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode inconnu : {mode}")
        self.cfg = config or ClusterConfig()
        self.mode = mode
        self.series = [(np.asarray(ts), np.asarray(d, dtype=np.float64)) for ts, d in series]
        self.feats = [hour_features(ts) for ts, _ in self.series]
        self.episode_hours = episode_hours
        n_actions = len(self.cfg.actions) if mode == "delta" else len(self.cfg.headrooms)
        self.action_space = spaces.Discrete(n_actions)
        # 8 niveaux de demande/capacité, 2 erreurs de prévision, 2 signaux de budget, 4 calendaires
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.OBS_DIM,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.k, self.t, self.end, self.instances = 0, self.LAGS, self.LAGS + 1, 1
        self.window_start, self.degraded_hours = self.LAGS, 0

    @classmethod
    def from_history(cls, history, last_hour: np.datetime64, instances: int, config: ClusterConfig | None = None,
                     mode: str = "headroom", degraded_hours: int = 0, hours_into_window: int = 0):
        """Environnement « figé » sur l'instant présent, à partir de l'historique récent.

        Utilisé en production (API) : on reconstruit exactement l'observation vue à
        l'entraînement, avec le même code, donc sans écart entre entraînement et service.
        """
        h = np.asarray(history, dtype=np.float64)[-(cls.LAGS + 1):]
        if len(h) < cls.LAGS + 1:
            raise ValueError(f"il faut au moins {cls.LAGS + 1} heures d'historique")
        last = np.datetime64(last_hour, "h")
        ts = last - np.arange(len(h) - 1, -2, -1).astype("timedelta64[h]")  # inclut l'heure suivante
        env = cls([(ts, np.append(h, 0.0))], config, episode_hours=None, mode=mode)
        env.k, env.t = 0, len(h) - 1
        env.instances = env.clip(int(instances))
        env.window_start = env.t - int(hours_into_window)
        env.degraded_hours = int(degraded_hours)
        return env

    # ------------------------------------------------------------------ outils
    @property
    def demand(self) -> np.ndarray:
        return self.series[self.k][1]

    def needed(self, d: float) -> int:
        """Instances nécessaires pour tenir `d` sous le seuil d'utilisation."""
        return int(np.ceil(d / self.cfg.target_util))

    def clip(self, n: int) -> int:
        return int(np.clip(n, self.cfg.min_instances, self.cfg.max_instances))

    def budget_left(self) -> float:
        """Heures dégradées encore autorisées dans la fenêtre (négatif = budget dépassé)."""
        return self.cfg.budget_hours - self.degraded_hours

    def window_progress(self) -> float:
        return (self.t - self.window_start) / self.cfg.window_hours

    def forecast(self, t: int | None = None) -> float:
        """Prévision saisonnière de la demande en t+1 : « même heure hier », corrigée du niveau du jour."""
        d = self.demand
        t = self.t if t is None else t
        return float(d[t + 1 - 24] * d[t] / max(1e-9, d[t - 24]))

    def forecast_errors(self, window: int = 24) -> np.ndarray:
        """Erreurs relatives des `window` dernières prévisions (réel / prévu − 1)."""
        d, t = self.demand, self.t
        idx = np.arange(t - window + 1, t + 1)
        pred = d[idx - 24] * d[idx - 1] / np.maximum(1e-9, d[idx - 25])
        return d[idx] / np.maximum(1e-9, pred) - 1.0

    def action_for_target(self, target: int) -> int:
        """Plus petite action (mode delta) qui atteint au moins `target`, sinon la plus grande."""
        if self.mode != "delta":
            raise RuntimeError("action_for_target n'a de sens qu'en mode 'delta'")
        target = self.clip(target)
        reach = [self.clip(self.instances + d) for d in self.cfg.actions]
        ok = [i for i, n in enumerate(reach) if n >= target]
        if not ok:
            return int(np.argmax(reach))
        return min(ok, key=lambda i: reach[i])

    def target_for_action(self, action: int) -> int:
        if self.mode == "delta":
            return self.clip(self.instances + self.cfg.actions[int(action)])
        return self.clip(self.needed(self.forecast() * (1 + self.cfg.headrooms[int(action)])))

    def _obs(self) -> np.ndarray:
        d, t, m = self.demand, self.t, float(self.cfg.max_instances)
        err = self.forecast_errors()
        levels = np.array([d[t], d[t - 1], d[t - 2], d[t - 24], d[t + 1 - 24], d[t + 1 - 168],
                           self.forecast(), self.instances], dtype=np.float32) / m
        budget = np.array([np.clip(self.budget_left() / self.cfg.budget_hours, -1, 1), self.window_progress()], dtype=np.float32)
        errs = np.array([np.clip(err[-1], -1, 1), np.clip(np.abs(err).mean(), 0, 1)], dtype=np.float32)
        return np.concatenate([levels, errs, budget, self.feats[self.k][t + 1]]).astype(np.float32)

    # ------------------------------------------------------------------ API gym
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        self.k = options.get("series", int(self._rng.integers(len(self.series))))
        n = len(self.demand)
        if self.episode_hours is None:
            self.t, self.end = self.LAGS, n - 1
        else:
            self.t = int(self._rng.integers(self.LAGS, n - 1 - self.episode_hours))
            self.end = self.t + self.episode_hours
        self.window_start, self.degraded_hours = self.t, 0
        self.instances = self.clip(self.needed(self.demand[self.t]))
        return self._obs(), {}

    def step(self, action: int):
        cfg = self.cfg
        new = self.target_for_action(action)
        started = max(0, new - self.instances)
        self.instances = new
        self.t += 1
        if self.t - self.window_start > cfg.window_hours:  # nouvelle fenêtre de SLO
            self.window_start, self.degraded_hours = self.t - 1, 0
        d = float(self.demand[self.t])
        infra = new * cfg.price_per_hour + started * cfg.boot_cost
        degraded = d > cfg.target_util * new
        self.degraded_hours += degraded
        over = degraded and self.degraded_hours > cfg.budget_hours
        unserved = max(0.0, d - new)
        sla = over * cfg.over_budget_penalty + unserved * cfg.drop_cost
        info = StepInfo(d, new, infra, sla, degraded, over, unserved)
        shaping = cfg.budget_shadow_price * (degraded and not over)
        reward = -(infra + sla + shaping) * cfg.reward_scale
        terminated = self.t >= self.end
        obs = self._obs() if not terminated else np.zeros(self.OBS_DIM, np.float32)
        return obs, reward, terminated, False, {"step": info}


@dataclass
class EpisodeReport:
    """Métriques d'un épisode complet, en langage métier."""

    hours: int = 0
    infra_cost: float = 0.0
    sla_cost: float = 0.0
    degraded_hours: int = 0
    over_budget_hours: int = 0
    unserved: float = 0.0
    demand: float = 0.0
    instance_hours: float = 0.0
    trace: list = field(default_factory=list)

    FIELDS = ("hours", "infra_cost", "sla_cost", "degraded_hours", "over_budget_hours", "unserved", "demand", "instance_hours")

    def add(self, s: StepInfo, keep_trace: bool) -> None:
        self.hours += 1
        self.infra_cost += s.infra_cost
        self.sla_cost += s.sla_cost
        self.degraded_hours += s.degraded
        self.over_budget_hours += s.over_budget
        self.unserved += s.unserved
        self.demand += s.demand
        self.instance_hours += s.instances
        if keep_trace:
            self.trace.append((s.demand, s.instances))

    def merge(self, other: EpisodeReport) -> None:
        for f in self.FIELDS:
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @property
    def total_cost(self) -> float:
        return self.infra_cost + self.sla_cost

    def summary(self) -> dict:
        return {
            "total_cost": round(self.total_cost, 2),
            "infra_cost": round(self.infra_cost, 2),
            "sla_cost": round(self.sla_cost, 2),
            "healthy_pct": round(100 * (1 - self.degraded_hours / max(1, self.hours)), 2),
            "over_budget_hours": int(self.over_budget_hours),
            "unserved_pct": round(100 * self.unserved / max(1e-9, self.demand), 3),
            "avg_util_pct": round(100 * self.demand / max(1e-9, self.instance_hours), 1),
        }
