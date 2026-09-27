"""Trafic réel : pages vues horaires de Wikipédia (Wikimedia Pageviews API, données CC0).

Chaque série devient une charge de travail pour un cluster simulé : le nombre de
pages vues par heure est converti en « instances nécessaires » (voir `to_demand`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
    "{project}/all-access/user/hourly/{start}/{end}"
)
USER_AGENT = "scalemind/0.1 (portfolio project; educational use)"
PROJECTS = ["fr.wikipedia", "en.wikipedia", "de.wikipedia", "es.wikipedia", "ja.wikipedia"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# Découpage temporel : on entraîne sur 2023, on évalue sur 2024 (jamais vu à l'entraînement).
TRAIN = ("2023-01-01", "2023-12-31 23:00")
TEST = ("2024-01-01", "2024-12-31 23:00")


def _fetch_month(session: requests.Session, project: str, month: pd.Timestamp) -> list[dict]:
    start = month.strftime("%Y%m%d00")
    end = (month + pd.offsets.MonthEnd(0)).strftime("%Y%m%d23")
    url = API.format(project=project, start=start, end=end)
    for attempt in range(6):
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()["items"]
        # 429 = trop de requêtes : on respecte le délai demandé, sinon backoff exponentiel
        wait = float(r.headers.get("Retry-After", 0) or 0) or 5 * 2**attempt
        time.sleep(min(wait, 120))
    r.raise_for_status()
    return []


def download(project: str, start: str = "2023-01-01", end: str = "2024-12-31", force: bool = False) -> Path:
    """Télécharge une série horaire mois par mois et l'enregistre en CSV (mise en cache)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{project}.csv"
    if path.exists() and not force:
        return path
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    rows: list[dict] = []
    for month in pd.date_range(start, end, freq="MS"):
        rows += _fetch_month(session, project, month)
        time.sleep(1.5)  # politesse envers l'API (limite de débit)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
    df[["timestamp", "views"]].to_csv(path, index=False)
    return path


def load_series(project: str) -> pd.Series:
    """Série horaire continue (les rares heures manquantes sont interpolées)."""
    path = RAW_DIR / f"{project}.csv"
    if not path.exists():
        download(project)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    s = df.set_index("timestamp")["views"].astype(float).sort_index()
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(full).interpolate(limit_direction="both").rename(project)


def to_demand(views: pd.Series, reference: pd.Series, peak_instances: float = 20.0) -> pd.Series:
    """Convertit des pages vues en instances nécessaires.

    On fixe la capacité d'une instance pour que le 99e centile du trafic de référence
    (la période d'entraînement) demande `peak_instances` instances. Même échelle
    pour le train et le test, donc aucune fuite d'information du futur.
    """
    capacity_per_instance = reference.quantile(0.99) / peak_instances
    return views / capacity_per_instance


def split(project: str, peak_instances: float = 20.0) -> tuple[pd.Series, pd.Series]:
    """Demande (en instances) pour la période d'entraînement et de test."""
    s = load_series(project)
    train, test = s[TRAIN[0]:TRAIN[1]], s[TEST[0]:TEST[1]]
    return to_demand(train, train, peak_instances), to_demand(test, train, peak_instances)


def main() -> None:
    p = argparse.ArgumentParser(description="Télécharge le trafic Wikipédia horaire.")
    p.add_argument("--projects", nargs="*", default=PROJECTS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    for project in args.projects:
        path = download(project, force=args.force)
        s = load_series(project)
        print(f"{project:14s} {len(s):6d} heures  {s.index.min()} -> {s.index.max()}  ({path.name})")


if __name__ == "__main__":
    main()
