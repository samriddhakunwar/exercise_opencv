"""
progress.py
-----------
Workout progress report from the history database.

Usage::

    python progress.py                 # summary + charts in data/charts/
    python progress.py --no-charts     # summary only
    python progress.py --db other.db --out some/dir

Prints totals, the current daily streak and personal bests per exercise,
then saves PNG charts: reps per week, average squat depth over time and
plank hold time over time.
"""

from __future__ import annotations

import argparse
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import config
import history
from exercises import EXERCISES

# Chart palette (validated categorical order, light surface)
_SURFACE   = "#fcfcfb"
_INK       = "#0b0b0b"
_INK_2     = "#52514e"
_MUTED     = "#898781"
_GRID      = "#e1e0d9"
_BASELINE  = "#c3c2b7"
_SERIES    = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _date(row) -> datetime.date:
    return datetime.datetime.fromisoformat(row["date"]).date()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def current_streak(dates: Iterable[datetime.date], today: Optional[datetime.date] = None) -> int:
    """Consecutive days with a workout, ending today (or yesterday).

    A streak isn't broken until a full day is missed, so a streak that
    ended yesterday still counts until today is over.
    """
    days  = set(dates)
    today = today or datetime.date.today()
    day   = today if today in days else today - datetime.timedelta(days=1)
    streak = 0
    while day in days:
        streak += 1
        day -= datetime.timedelta(days=1)
    return streak


def personal_bests(rows: Sequence) -> Dict[str, Dict[str, float]]:
    """Per exercise: most reps in a session and best best_metric."""
    bests: Dict[str, Dict[str, float]] = {}
    for row in rows:
        key = row["exercise"]
        if key not in EXERCISES:
            continue
        b = bests.setdefault(key, {"reps": 0})
        b["reps"] = max(b["reps"], row["reps"])
        value = row["best_metric"]
        if value is None:
            continue
        lower = EXERCISES[key].factory.BEST_LOWER_IS_BETTER
        if "best" not in b or (value < b["best"] if lower else value > b["best"]):
            b["best"] = value
    return bests


def summary_text(rows: Sequence, today: Optional[datetime.date] = None) -> str:
    """Human-readable progress summary."""
    if not rows:
        return "No workouts recorded yet — finish a session in app.py first."

    lines = [
        "WORKOUT PROGRESS",
        "=" * 40,
        f"Sessions        : {len(rows)}",
        f"Total reps      : {sum(r['reps'] for r in rows if r['exercise'] != 'plank')}",
        f"Total time      : {sum(r['duration'] for r in rows) / 60:.0f} min",
        f"Calories        : {sum(r['calories'] for r in rows):.0f} kcal",
        f"Current streak  : {current_streak((_date(r) for r in rows), today)} day(s)",
        "",
        "PERSONAL BESTS",
        "-" * 40,
    ]
    bests = personal_bests(rows)
    for key, spec in EXERCISES.items():
        if key not in bests:
            continue
        b = bests[key]
        parts = [] if key == "plank" else [f"{b['reps']} reps"]
        if "best" in b:
            parts.append(f"{spec.best_label.lower()} {spec.best_fmt.format(b['best'])}")
        lines.append(f"{spec.name:<14}: {', '.join(parts)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style(ax, title: str, ylabel: str) -> None:
    """Recessive axes: horizontal hairline grid, baseline only, muted ticks."""
    ax.set_facecolor(_SURFACE)
    ax.set_title(title, loc="left", color=_INK, fontsize=13, pad=12)
    ax.set_ylabel(ylabel, color=_INK_2)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED, length=0)


def _line_chart(ax, rows: List, value_key: str) -> None:
    dates  = [datetime.datetime.fromisoformat(r["date"]) for r in rows]
    values = [r[value_key] for r in rows]
    ax.plot(dates, values, color=_SERIES[0], linewidth=2, marker="o", markersize=8,
            markeredgecolor=_SURFACE, markeredgewidth=2)
    ax.annotate(f"{values[-1]:.0f}", (dates[-1], values[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=_INK_2)


def save_charts(rows: Sequence, out_dir: str = config.CHARTS_DIR) -> List[Path]:
    """Write progress charts as PNGs; returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def finish(fig, name: str) -> None:
        fig.axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.autofmt_xdate()
        fig.tight_layout()
        path = out / name
        fig.savefig(path, dpi=120, facecolor=_SURFACE)
        plt.close(fig)
        written.append(path)

    # ---- Reps per week, stacked by exercise ----------------------------
    rep_keys = [k for k in EXERCISES if k != "plank"]
    weekly: Dict[datetime.date, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["exercise"] in rep_keys:
            d = _date(r)
            weekly[d - datetime.timedelta(days=d.weekday())][r["exercise"]] += r["reps"]
    if weekly:
        weeks = sorted(weekly)
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=_SURFACE)
        bottom = [0] * len(weeks)
        for i, key in enumerate(rep_keys):      # fixed order → fixed colour per exercise
            vals = [weekly[w].get(key, 0) for w in weeks]
            if not any(vals):
                continue
            ax.bar(weeks, vals, width=4, bottom=bottom, color=_SERIES[i],
                   edgecolor=_SURFACE, linewidth=1.5, label=EXERCISES[key].name.title())
            bottom = [b + v for b, v in zip(bottom, vals)]
        for w, total in zip(weeks, bottom):
            ax.annotate(str(total), (w, total), xytext=(0, 4), textcoords="offset points",
                        ha="center", color=_INK_2, fontsize=9)
        _style(ax, "Reps per week", "Reps")
        ax.legend(frameon=False, labelcolor=_INK_2, loc="upper left",
                  bbox_to_anchor=(1.0, 1.0))
        finish(fig, "reps_per_week.png")

    # ---- Average squat depth over time --------------------------------
    squats = [r for r in rows if r["exercise"] == "squat" and r["avg_metric"] is not None]
    if squats:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=_SURFACE)
        _line_chart(ax, squats, "avg_metric")
        _style(ax, "Average squat depth (lower angle = deeper)", "Knee angle (°)")
        finish(fig, "squat_depth.png")

    # ---- Plank hold over time -----------------------------------------
    planks = [r for r in rows if r["exercise"] == "plank" and r["best_metric"] is not None]
    if planks:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=_SURFACE)
        _line_chart(ax, planks, "best_metric")
        _style(ax, "Longest plank hold per session", "Seconds")
        ax.set_ylim(bottom=0)
        finish(fig, "plank_hold.png")

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the summary and write charts."""
    parser = argparse.ArgumentParser(description="Workout progress report")
    parser.add_argument("--db", default=config.HISTORY_DB, help="history database path")
    parser.add_argument("--out", default=config.CHARTS_DIR, help="chart output folder")
    parser.add_argument("--no-charts", action="store_true", help="skip chart generation")
    args = parser.parse_args(argv)

    rows = history.all_sessions(args.db)
    print(summary_text(rows))
    if rows and not args.no_charts:
        paths = save_charts(rows, args.out)
        print("\nCharts:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
