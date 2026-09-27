from scalemind.env import AutoscalingEnv
from scalemind.evaluate import evaluate
from scalemind.policies import BudgetAwarePolicy, OraclePolicy, ReactivePolicy, SeasonalPolicy, StaticPolicy
from scalemind.qlearning import QLearningAgent


def _cost(policy, series, mode="delta"):
    env = AutoscalingEnv(series, episode_hours=None, mode=mode)
    return evaluate(policy, env, ["synthetic"])["total"]


def test_oracle_never_degrades_on_smooth_traffic(series):
    assert _cost(OraclePolicy(), series)["healthy_pct"] > 99.5


def test_forecasting_beats_static_and_reactive(series):
    seasonal = _cost(SeasonalPolicy(0.05), series)["total_cost"]
    assert seasonal < _cost(StaticPolicy(25), series)["total_cost"]
    assert seasonal < _cost(ReactivePolicy(0.7), series)["total_cost"]


def test_budget_rule_runs(series):
    report = _cost(BudgetAwarePolicy(0.0, 0.2), series)
    assert report["total_cost"] > 0


def test_qlearning_learns_something(series):
    env = AutoscalingEnv(series, episode_hours=240, seed=0, mode="headroom")
    agent = QLearningAgent(env.action_space.n, mode="headroom", eps_decay_steps=5_000, seed=0)
    curve = agent.train(env, 10_000, log_every=1_000)
    assert curve[-1][1] > curve[0][1]  # le retour moyen s'améliore
    assert agent.q.any()
