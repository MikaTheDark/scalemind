"""Graphiques du README, générés à partir de results/metrics.json et web/data/replay.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1d2230", "#6b7185", "#e6e8ee"
COLORS = {  # couleur = identité de la politique, identique dans tous les graphiques
    "Statique (pic)": "#9aa0b0",
    "Réactif (HPA)": "#3f7fd9",
    "Prévision saisonnière": "#1f9c83",
    "Prévision + règle SRE": "#5b8c3a",
    "Q-learning (marge)": "#c255bd",
    "DQN (marge)": "#8a6fd1",
    "PPO (±instances)": "#e0a95a",
    "PPO (marge)": "#c4850f",
    "Oracle (connaît le futur)": "#4a4f5e",
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def cost_bars(results: dict, out: Path) -> None:
    names = list(results)
    infra = [results[n]["total"]["infra_cost"] / 1000 for n in names]
    sla = [results[n]["total"]["sla_cost"] / 1000 for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    y = range(len(names))
    ax.barh(y, infra, color=[COLORS.get(n, "#888") for n in names], height=0.62, label="Infrastructure")
    ax.barh(y, sla, left=infra, color="#ef5f55", height=0.62, label="Pénalités SLO")
    for i, n in enumerate(names):
        t = results[n]["total"]
        ax.text(infra[i] + sla[i] + 1.5, i, f"{t['total_cost'] / 1000:.1f} k€  ·  saines {t['healthy_pct']:.1f} %",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), names, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Coût total sur 2024, 5 Wikipédias (k€)", color=MUTED, fontsize=9)
    ax.set_xlim(0, max(a + b for a, b in zip(infra, sla)) * 1.32)
    _style(ax)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Coût annuel = infrastructure + pénalités SLO (plus court = mieux)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def learning_curves(curves: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    for name, pts in curves.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot([x / 1000 for x in xs], ys, color=COLORS.get(name, "#888"), linewidth=2, label=name)
    ax.set_xlabel("Heures simulées vues à l'entraînement (milliers)", color=MUTED, fontsize=9)
    ax.set_ylabel("Récompense par épisode\n(2 semaines, plus haut = mieux)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Courbes d'apprentissage sur le trafic 2023", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def week_trace(replay: dict, out: Path, series: str, policies: list[str], start: int = 0, hours: int = 168) -> None:
    s = replay["series"][series]
    target = replay["config"]["target_util"]
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=150)
    x = range(hours)
    need = [d / target for d in s["demand"][start:start + hours]]
    ax.fill_between(x, need, color="#d9dce5", step="mid", label="Besoin réel (trafic / 85 %)")
    for p in policies:
        if p in s["instances"]:
            ax.step(x, s["instances"][p][start:start + hours], where="mid", color=COLORS.get(p, "#888"), linewidth=1.8, label=p)
    ticks = list(range(0, hours, 24))
    ax.set_xticks(ticks, [s["timestamps"][start + t][:10] for t in ticks], fontsize=8)
    ax.set_ylabel("Instances", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=len(policies) + 1, loc="upper left", bbox_to_anchor=(0, 1.16))
    ax.set_title(f"Une semaine réelle de 2024 ({series}) : capacité vs besoin", loc="left", fontsize=11, color=INK, pad=28)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


SWEEP_LABELS = {
    "q_g999": "Q-learning tabulaire, γ=0,999",
    "q_g99_sh3": "Q-learning tabulaire, γ=0,99 + prix fictif 3 €",
    "dqn_g999": "DQN, γ=0,999",
    "dqn_g99_sh3": "DQN, γ=0,99 + prix fictif 3 €",
    "ppo_g999": "PPO, γ=0,999",
    "ppo_g995": "PPO, γ=0,995",
    "ppo_g999_sh2": "PPO, γ=0,999 + prix fictif 2 €",
    "ppo_g999_sh2_s1": "PPO, γ=0,999 + prix fictif 2 € (graine 2)",
    "ppo_g99_sh4": "PPO, γ=0,99 + prix fictif 4 € (1 M pas)",
    "final_sh4": "PPO, γ=0,99 + prix fictif 4 € (2 M pas)",
    "final_sh6": "PPO, γ=0,99 + prix fictif 6 € (2 M pas)",
    "ablation_delta": "Ablation : PPO sans prévision (±instances)",
}


def sweep_bars(sweep: dict, results: dict, out: Path) -> None:
    fams = sorted(sweep, key=lambda f: sweep[f]["cost_mean"])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(fams) + 1.4), dpi=150)
    y = range(len(fams))
    means = [sweep[f]["cost_mean"] / 1000 for f in fams]
    stds = [sweep[f]["cost_std"] / 1000 if sweep[f]["runs"] > 1 else 0 for f in fams]
    # points + barres d'erreur : l'axe ne part pas de zéro, des barres pleines exagéreraient les écarts
    for yi, f, mu, sd in zip(y, fams, means, stds):
        col = "#c4850f" if ("ppo" in f or "final" in f or "ablation" in f) else "#8a6fd1" if "dqn" in f else "#c255bd"
        ax.errorbar(mu, yi, xerr=sd or None, fmt="o", color=col, markersize=8, elinewidth=1.5, capsize=4)
        ax.text(mu + max(sd, 0) + 0.4, yi, f"{mu:.1f} k€", va="center", fontsize=8, color=INK)
    labels = [SWEEP_LABELS.get(f, f) + (f"  ({sweep[f]['runs']} graines)" if sweep[f]["runs"] > 1 else "") for f in fams]
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    for name, style in (("Prévision + règle SRE", "--"), ("Oracle (connaît le futur)", ":")):
        if name in results:
            v = results[name]["total"]["total_cost"] / 1000
            ax.axvline(v, color="#5b8c3a" if "SRE" in name else "#4a4f5e", linestyle=style, linewidth=1.5)
            ax.text(v, -0.9, name.split(" (")[0] if "Oracle" in name else "Règle SRE", fontsize=8, color=MUTED, ha="center")
    lo = min(min(means), results.get("Oracle (connaît le futur)", {"total": {"total_cost": 1e9}})["total"]["total_cost"] / 1000)
    ax.set_xlim(lo * 0.95, max(m + s for m, s in zip(means, stds)) * 1.02)
    ax.set_xlabel("Coût total 2024, 5 Wikipédias (k€), moyenne ± écart-type entre graines", color=MUTED, fontsize=9)
    ax.set_title("Expériences : horizon (γ) et prix fictif du budget d'erreur", loc="left", fontsize=11, color=INK, pad=18)
    _style(ax)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_plots(results_dir: Path) -> None:
    m = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    cost_bars(m["results"], results_dir / "cost_comparison.png")
    if m.get("curves"):
        learning_curves(m["curves"], results_dir / "learning_curves.png")
    if m.get("sweep"):
        sweep_bars(m["sweep"], m["results"], results_dir / "experiments.png")
    replay_path = results_dir.parent / "web" / "data" / "replay.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        pols = [p for p in ("Réactif (HPA)", "Prévision + règle SRE", "PPO (marge)") if p in m["results"]]
        week_trace(replay, results_dir / "week_trace.png", next(iter(replay["series"])), pols, start=168)


if __name__ == "__main__":
    make_plots(Path(__file__).resolve().parent.parent / "results")
