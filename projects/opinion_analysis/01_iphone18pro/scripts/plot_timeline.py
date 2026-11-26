"""Plot a successful daily global TimelineVolRaw response; never plot HTTP failures as zero."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from gdelt_probe import ROOT, utc


def read_daily(run: Path) -> tuple[list[datetime], list[float], list[float], str]:
    if not (run / "pilot.sqlite").is_file():
        raise ValueError("No pilot.sqlite found in the supplied run")
    connection = sqlite3.connect((run / "pilot.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        request = connection.execute(
            "SELECT request_id, start_utc, end_exclusive_utc FROM requests "
            "WHERE query_id='base__global' AND mode='TimelineVolRaw' AND status='ok' "
            "ORDER BY requested_at DESC LIMIT 1"
        ).fetchone()
        if request is None:
            raise ValueError("No successful global TimelineVolRaw response; no chart created")
        request_id, start, end = request
        rows = connection.execute(
            "SELECT series, date_raw, value, norm FROM timeline_points WHERE request_id=? ORDER BY date_raw",
            (request_id,),
        ).fetchall()
    finally:
        connection.close()
    if not rows or len({row[0] for row in rows}) != 1:
        raise ValueError("Expected exactly one nonempty volume series; inspect the raw response")
    points = {utc(row[1]): (float(row[2]), row[3]) for row in rows}
    if any(date.hour or date.minute or date.second for date in points):
        raise ValueError("Expected daily UTC bins; do not relabel intraday points as daily counts")
    dates, counts, percentages = [], [], []
    day, stop = utc(start), utc(end)
    if day.hour or day.minute or day.second or stop.hour or stop.minute or stop.second:
        raise ValueError("Plot only complete UTC days")
    while day < stop:
        count, norm = points.get(day, (math.nan, None))
        if math.isfinite(count) and count < 0:
            raise ValueError("A volume count cannot be negative")
        dates.append(day)
        counts.append(count)
        percentages.append(100.0 * count / norm if norm is not None and norm > 0 else math.nan)
        day += timedelta(days=1)
    if not any(math.isfinite(value) for value in counts):
        raise ValueError("No response points within the requested daily window")
    return dates, counts, percentages, request_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        dates, counts, percentages, request_id = read_daily(args.run)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    # Keep plotting-tool caches inside the project.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    config = json.loads((args.run / "config_snapshot.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), layout="constrained")
    axes[0].plot(dates, counts, color="#2463a6", marker=".")
    axes[1].plot(dates, percentages, color="#177e70", marker=".")
    axes[0].set_ylabel("GDELT matched records")
    axes[1].set_ylabel("% of GDELT total coverage")
    axes[0].set_title("iPhone 18 Pro: global news coverage (UTC)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for event in config["events"]:
        when = utc(event["date"] + "T00:00:00Z")
        if dates[0] <= when <= dates[-1]:
            for axis in axes:
                axis.axvline(when, color="#888888", linestyle="--", alpha=0.6)
            axes[0].text(when, 1.01, event["label"], transform=axes[0].get_xaxis_transform(), fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].set_xlabel("Missing or invalid bins remain gaps; these are not public approval rates.")
    if not any(math.isfinite(value) for value in percentages):
        axes[1].text(0.5, 0.5, "No valid norm denominator returned", transform=axes[1].transAxes, ha="center")
    output = args.run / "figures"
    output.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"global_volume_{request_id}.{extension}", dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
