import numpy as np
import pytest

from tests.conftest import synthetic_series

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import scalemind.serve as serve  # noqa: E402


@pytest.fixture
def client():
    return TestClient(serve.app)


def _payload(hours: int):
    ts, d = synthetic_series(days=10)
    return {
        "load_history": (d[-hours:] * 1000).tolist(),
        "capacity_per_instance": 1000,
        "current_instances": 12,
        "last_hour": str(ts[-1]) + ":00:00",
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_recommend_stays_within_bounds(client):
    r = client.post("/v1/recommend", json=_payload(200))
    assert r.status_code == 200
    body = r.json()
    assert serve.cfg.min_instances <= body["target_instances"] <= serve.cfg.max_instances
    assert body["delta"] == body["target_instances"] - 12


def test_short_history_falls_back_to_hold(client):
    body = client.post("/v1/recommend", json=_payload(50)).json()
    assert body["fallback"] is True and body["delta"] == 0


def test_missing_model_falls_back_to_forecast(client, monkeypatch):
    monkeypatch.setattr(serve, "model", None)
    body = client.post("/v1/recommend", json=_payload(200)).json()
    assert body["fallback"] is True and body["policy"] == "Prévision + règle SRE"


def test_numpy_export_matches_pytorch(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from scalemind.env import AutoscalingEnv

    env = AutoscalingEnv([synthetic_series()], episode_hours=100, seed=0)
    model = PPO("MlpPolicy", env, seed=0, policy_kwargs={"net_arch": [16, 16]})
    serve.export_sb3(model, "test", tmp_path / "p.npz", "headroom")
    pol = serve.NumpyPolicy(tmp_path / "p.npz")
    obs = np.stack([env.observation_space.sample() for _ in range(32)]).clip(-2, 2)
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(obs))
        ref = dist.distribution.logits.numpy()
    got = np.stack([pol.logits(o) for o in obs])
    # softmax invariant : on compare les logits centrés
    np.testing.assert_allclose(got - got.mean(1, keepdims=True), ref - ref.mean(1, keepdims=True), atol=1e-4)
