import numpy as np
import pytest


def synthetic_series(days: int = 45, seed: int = 0, base: float = 10.0, amp: float = 5.0, noise: float = 0.3):
    """Trafic synthétique à cycle journalier : les tests ne dépendent pas du réseau."""
    rng = np.random.default_rng(seed)
    n = days * 24
    ts = np.datetime64("2023-01-01T00", "h") + np.arange(n).astype("timedelta64[h]")
    hours = np.arange(n) % 24
    demand = base + amp * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(0, noise, n)
    return ts, np.maximum(demand, 0.5)


@pytest.fixture
def series():
    return [synthetic_series()]
