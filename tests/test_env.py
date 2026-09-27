import numpy as np
import pytest

from scalemind.env import AutoscalingEnv, ClusterConfig


def test_reset_gives_finite_observation(series):
    env = AutoscalingEnv(series, episode_hours=240, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (AutoscalingEnv.OBS_DIM,)
    assert np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("mode", AutoscalingEnv.MODES)
def test_episode_runs_to_the_end(series, mode):
    env = AutoscalingEnv(series, episode_hours=100, seed=1, mode=mode)
    env.reset()
    steps, done = 0, False
    while not done:
        _, r, done, _, info = env.step(env.action_space.sample())
        assert r <= 0  # la récompense est un coût négatif
        steps += 1
    assert steps == 100


def test_infra_cost_is_instances_times_price(series):
    cfg = ClusterConfig()
    env = AutoscalingEnv(series, cfg, episode_hours=50, seed=0, mode="delta")
    env.reset()
    before = env.instances
    _, _, _, _, info = env.step(cfg.actions.index(2))
    s = info["step"]
    assert s.instances == before + 2
    assert s.infra_cost == pytest.approx(s.instances * cfg.price_per_hour + 2 * cfg.boot_cost)


def test_baseline_actions_never_undershoot(series):
    env = AutoscalingEnv(series, episode_hours=50, seed=0, mode="delta")
    env.reset()
    for target in range(1, 30):
        n = env.target_for_action(env.action_for_target(target))
        max_reach = env.clip(env.instances + max(env.cfg.actions))
        assert n >= min(target, max_reach)


def test_penalty_only_once_error_budget_is_spent(series):
    # une seule instance face à une demande de ~10 : toutes les heures sont dégradées
    cfg = ClusterConfig(max_instances=1, drop_cost=0.0)
    env = AutoscalingEnv(series, cfg, episode_hours=200, seed=0, mode="delta")
    env.reset()
    over = []
    for _ in range(100):
        _, _, _, _, info = env.step(cfg.actions.index(0))
        over.append(info["step"].over_budget)
    first_penalty = over.index(True)
    assert first_penalty == int(np.floor(cfg.budget_hours))  # budget de 21,6 h → pénalité dès la 22e
    assert all(over[first_penalty:])


def test_error_budget_resets_every_window(series):
    cfg = ClusterConfig(max_instances=1, window_hours=48)
    env = AutoscalingEnv(series, cfg, episode_hours=None, mode="delta")
    env.reset()
    for _ in range(48):
        env.step(cfg.actions.index(0))
    assert env.degraded_hours == 48
    env.step(cfg.actions.index(0))
    assert env.degraded_hours == 1  # nouvelle fenêtre


def test_serving_rebuilds_exactly_the_training_observation(series):
    """Pas d'écart entraînement / production : même observation depuis l'historique brut."""
    env = AutoscalingEnv(series, episode_hours=None, mode="headroom")
    env.reset()
    for _ in range(300):
        env.step(4)
    ts, d = env.series[0]
    live = AutoscalingEnv.from_history(d[: env.t + 1], ts[env.t], env.instances, env.cfg, mode="headroom",
                                       degraded_hours=env.degraded_hours, hours_into_window=env.t - env.window_start)
    np.testing.assert_allclose(live._obs(), env._obs(), rtol=1e-6)


def test_from_history_requires_enough_hours():
    with pytest.raises(ValueError):
        AutoscalingEnv.from_history(np.ones(50), np.datetime64("2024-01-01T00"), 3)
