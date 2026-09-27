"""API de recommandation d'autoscaling (FastAPI).

    uvicorn scalemind.serve:app --port 8000

Choix d'architecture :
- L'entraînement utilise PyTorch, mais la politique est exportée en poids numpy
  (`models/policy.npz`) : l'image de service n'embarque pas PyTorch (image légère,
  démarrage rapide, inférence en moins d'une milliseconde).
- Garde-fous : bornes min/max, et repli automatique sur la règle « prévision + budget
  d'erreur SRE » si le modèle est absent ou si l'entrée est incomplète.
- L'API recommande, elle n'agit pas : un contrôleur (KEDA, HPA externe, script de
  scale set) applique la décision, ce qui permet un déploiement en « shadow mode ».
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .env import AutoscalingEnv, ClusterConfig
from .policies import BudgetAwarePolicy

MODEL_PATH = Path(os.environ.get("SCALEMIND_MODEL", Path(__file__).resolve().parent.parent / "models" / "policy.npz"))


class NumpyPolicy:
    """Réseau de neurones (MLP) exporté depuis stable-baselines3, exécuté en numpy."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        n = int(z["n_layers"])
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        self.activation = str(z["activation"])
        self.name = str(z["name"])
        self.mode = str(z["mode"])

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.tanh(x) if self.activation == "tanh" else np.maximum(x, 0)
        return x

    def reset(self, env):
        pass

    def act(self, env, obs):
        return int(np.argmax(self.logits(obs)))


def export_sb3(model, name: str, path: Path, mode: str) -> None:
    """Exporte l'acteur d'un modèle PPO ou le Q-réseau d'un DQN vers un .npz sans PyTorch."""
    import torch.nn as nn

    pol = model.policy
    if hasattr(pol, "q_net"):  # DQN
        mods, act = list(pol.q_net.q_net), "relu"
    else:  # PPO : extracteur de l'acteur puis tête d'action
        mods, act = list(pol.mlp_extractor.policy_net) + [pol.action_net], "tanh"
    linears = [m for m in mods if isinstance(m, nn.Linear)]
    arrays = {f"W{i}": m.weight.detach().numpy() for i, m in enumerate(linears)}
    arrays |= {f"b{i}": m.bias.detach().numpy() for i, m in enumerate(linears)}
    np.savez(path, n_layers=len(linears), activation=act, name=name, mode=mode, **arrays)


class RecommendRequest(BaseModel):
    load_history: list[float] = Field(..., description="Charge horaire observée (ex. requêtes/heure), la plus récente en dernier. Au moins 193 valeurs (8 jours).")
    capacity_per_instance: float = Field(..., gt=0, description="Charge qu'une instance absorbe en une heure")
    current_instances: int = Field(..., ge=0)
    last_hour: datetime = Field(..., description="Horodatage de la dernière valeur de l'historique")
    degraded_hours_in_window: int = Field(0, ge=0, description="Heures dégradées déjà consommées dans la fenêtre SLO en cours")
    hours_into_window: int = Field(0, ge=0, description="Heures écoulées depuis le début de la fenêtre SLO (30 jours)")


class RecommendResponse(BaseModel):
    target_instances: int
    delta: int
    policy: str
    fallback: bool
    reason: str
    latency_ms: float


cfg = ClusterConfig()
fallback_policy = BudgetAwarePolicy(low=0.03, high=0.12)
def load_model(path: Path) -> NumpyPolicy | None:
    """Charge la politique ; None si absente ou incompatible avec l'environnement actuel."""
    try:
        pol = NumpyPolicy(path)
    except (FileNotFoundError, OSError, KeyError):
        return None
    if pol.layers[0][0].shape[1] != AutoscalingEnv.OBS_DIM:
        # modèle entraîné sur une ancienne version des observations : on refuse de le servir
        return None
    return pol


model = load_model(MODEL_PATH)

app = FastAPI(title="ScaleMind", version=__version__, description="Recommandations d'autoscaling apprises par reinforcement learning.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None, "model": model.name if model else None}


@app.post("/v1/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    demand = np.asarray(req.load_history, dtype=np.float64) / req.capacity_per_instance
    if len(demand) < AutoscalingEnv.LAGS + 1 or not np.all(np.isfinite(demand)):
        # historique insuffisant : décision prudente, on garde la taille actuelle
        return RecommendResponse(target_instances=max(cfg.min_instances, req.current_instances), delta=0,
                                 policy="hold", fallback=True, reason=f"historique incomplet ({AutoscalingEnv.LAGS + 1} heures requises)",
                                 latency_ms=round((time.perf_counter() - t0) * 1000, 3))
    policy, fallback, reason = (model, False, "politique RL") if model else (fallback_policy, True, "modèle absent : prévision + règle SRE")
    mode = model.mode if model else "delta"
    env = AutoscalingEnv.from_history(demand, np.datetime64(req.last_hour.replace(tzinfo=None), "h"), req.current_instances, cfg,
                                      mode=mode, degraded_hours=req.degraded_hours_in_window,
                                      hours_into_window=min(req.hours_into_window, cfg.window_hours))
    target = env.target_for_action(policy.act(env, env._obs()))
    return RecommendResponse(target_instances=target, delta=target - req.current_instances, policy=getattr(policy, "name", "rl"),
                             fallback=fallback, reason=reason, latency_ms=round((time.perf_counter() - t0) * 1000, 3))
