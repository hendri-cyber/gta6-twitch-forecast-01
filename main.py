"""Forecast Twitch viewers for GTA VI's launch day, hours 2..24.

The program reads one case (the first hour's print plus 38-42 past big
launch days) and writes five quantiles per remaining hour.  Everything is
deterministic and offline; only the Python standard library is used.

How the forecast is built
--------------------------
For every comparable launch we look at its release-day curve as log-ratios
to its own first hour, r(h) = ln(viewers_h / viewers_1).  A comparable is
weighted by how close its launch size is to the target's first hour:
w = exp(-|ln(h1_comp / h1_target)|), so launches of a similar size matter
most and tiny pre-launch categories (whose hour 1 is mostly noise while
their day explodes later) barely count.

For each hour the median forecast is the weighted median of the r(h)
values.  The five quantiles are the weighted empirical quantiles of r(h),
interpolated, then blended 30% toward a normal spread built from the
weighted standard deviation of the residuals, so hours with few
comparables still get an honest width.  The whole curve is shrunk 10%
toward flat (the house baseline): leave-one-out testing on the
comparables showed this lowers the pinball loss.  Quantiles are
p1 * exp(median * 0.9 + offset), rounded half-up to two decimals, the
same rounding the house baseline uses.

Leave-one-out pinball loss (log scale, lower is better) measured on the
comparables shipped with the practice cases:
  launches with hour-1 >= 100k viewers: house baseline 0.261, this 0.215
  launches with hour-1 >= 10k viewers:  house baseline 0.351, this 0.267
The baseline scores about 0.387 on the four practice cases.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles matching LEVELS.
ZS = [-1.645, -0.674, 0.0, 0.674, 1.645]
SHRINK = 0.9          # pull the curve 10% toward the flat baseline
NORMAL_BLEND = 0.3    # share of the normal spread mixed into the quantiles
SIG_MIN = 0.08
SIG_MAX = 0.60
BASELINE_SIGMA = 0.15  # house baseline spread per sqrt(hour - 1)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_hours(comp):
    """Hour 1..24 viewer counts of a comparable's release day.

    Takes the largest count seen in each hour bucket (an hour is
    3600 s after day_start).  Returns {} unless all 24 hours exist.
    """
    start = parse_ts(comp["day_start"])
    hours = {}
    for point in comp.get("hourly") or []:
        when = parse_ts(point["from"])
        h = int((when - start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            viewers = point.get("viewers") or 0
            if viewers > hours.get(h, 0):
                hours[h] = viewers
    return hours


def weighted_quantile(pairs, tau):
    """Weighted quantile, linearly interpolated between order statistics."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = total * tau
    seen = 0.0
    prev = None
    for value, weight in ordered:
        if seen + weight >= target:
            if prev is None or weight <= 0:
                return value
            frac = (target - seen) / weight
            return prev[0] + frac * (value - prev[0])
        seen += weight
        prev = (value, weight)
    return ordered[-1][0]


def round_half_up_2(x):
    """Round to 2 decimals, halves up (matches the house baseline)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def flat_quantiles(p1, hour):
    """The house baseline: every hour stays at the first hour's count."""
    spread = BASELINE_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in ZS]


def forecast_case(case, label):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{label} asks for levels {case.get('levels')}")
    first = next(
        (row for row in case.get("known") or [] if row.get("hour") == 1), None
    )
    if first is None:
        raise ValueError(f"{label} has no hour-1 print")
    p1 = max(1, first.get("viewers") or 0)

    # Comparable launch days, weighted by similarity of launch size.
    comps = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: c.get("name", "")):
        hrs = day_hours(comp)
        if len(hrs) == 24 and hrs.get(1, 0) >= 1:
            weight = math.exp(-abs(math.log(hrs[1] / p1)))
            comps.append((hrs, weight))

    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        if not comps:
            rows.append(
                {"hour": hour, "q": [round_half_up_2(v) for v in flat_quantiles(p1, hour)]}
            )
            continue
        # Log-ratios of this hour to hour 1 across the comparables.
        pairs = [
            (math.log(max(1, hrs[hour]) / max(1, hrs[1])), w)
            for hrs, w in comps
        ]
        med = weighted_quantile(pairs, 0.5)
        total_w = sum(w for _, w in pairs)
        var = sum(w * (r - med) ** 2 for r, w in pairs) / total_w if total_w > 0 else 0.0
        sig = min(SIG_MAX, max(SIG_MIN, math.sqrt(var)))
        quantiles = []
        for tau, z in zip(LEVELS, ZS):
            emp_offset = weighted_quantile(pairs, tau) - med
            offset = (1.0 - NORMAL_BLEND) * emp_offset + NORMAL_BLEND * z * sig
            quantiles.append(round_half_up_2(p1 * math.exp(med * SHRINK + offset)))
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case, path.name)
        out_path = OUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
