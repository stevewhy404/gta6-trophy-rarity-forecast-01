"""GTA VI trophy rarity forecaster (Python 3.13, standard library only).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, five quantiles
(5/25/50/75/95%) of its Exophase earn rate at the case's target, from the
house's recorded reads in the case's history. Scored by pinball loss on the
logit scale, so the program writes the quantiles it really believes.

How it works
------------
A trophy's earn rate is holders / players-tracked, and both move: holders
only grow as players earn trophies, players-tracked only grows as more
profiles sync, and Exophase refreshes the page's counts on its own schedule
(hours apart), so the displayed rates sit flat between refreshes and jump
at them. The forecast therefore works with the rate's own growth curve
in log time, and never assumes a flat page means flat rates.

1. Distinct states. Consecutive identical reads are one refresh, so the
   history is collapsed to its distinct states (time, players, rates).
   Holders per trophy are recovered as round(rate * players / 100): the
   page prints rates to 2 decimals, so the count is exact up to rounding.

2. Trend mode (history shows real movement: >= 3 distinct states).
   Per trophy, log(rate) is fitted by least squares against
   ln(1 + hours since the first read) -- a power-law growth curve, the
   shape a launch's trophy rates follow -- and read at the target.
   The fit that matters is the last 24 hours' (the recent slope is
   what a saturating curve is doing now; the house curve baseline fits
   the last 24 hours the same way), with the all-history fit kept as
   a 15% anchor against a noisy recent window; slopes are clamped to
   +/-1.5 so no trophy's rate can be extrapolated explosively, and
   reads below 0.05% are left out of the fit (they carry almost no
   information and their log would anchor the slope). The implied
   logit move from the last rate is shrunk 30% toward the
   cross-trophy median move
   (rare trophies share one launch curve) and damped 10% for
   extrapolating past the observed range, then clamped to 2.5 logit
   units. The spread combines the fits' pooled residual -- which
   already contains whatever refresh jumps the trend did not explain
   -- and a random-walk term (0.5 logit per root-day, the house
   baseline's scale).

3. Refresh-prior mode (history is flat: < 3 distinct states, or nothing
   moved). No trend can be fitted, but a flat page does not mean the
   rates will stay flat: the next refresh brings the players who have
   synced since, and fresh joiners dilute every rate but the universal
   ones (competition.yaml documents the first refresh taking players
   638 -> 3,868 while every rate but the 100% one fell). So the forecast
   is a mixture over the refresh landing before the target or not:
   p_refresh grows with the time since the last distinct state plus the
   horizon, over a 6-hour refresh scale; on a refresh, players multiply
   by ~4 and a fresh joiner holds a trophy with probability beta of the
   incumbent rate (beta ~ 0.1 for rare trophies, -> 1 for trophies
   everyone holds). The 5% quantile sits at the diluted rate, the median
   at the probability-weighted centre, the 95% just above the last rate.

Everything is a deterministic function of the case file: no randomness,
no network, no clock. Two runs on the same case write identical bytes.

Contract (from the kit's RULES.txt): read every <id>.in in the input
directory (MAKEFUN_INPUT_DIR, else /input), write <id>.out to the output
directory (MAKEFUN_OUTPUT_DIR, else /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, one row
per trophy, quantiles non-decreasing in [0.01, 99.99], rounded to 4
decimals, at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (see the module docstring for where each comes from) ---
SIGMA_RW = 0.5          # logit random-walk vol per root-day (house baseline's)
REFRESH_SCALE_H = 6.0   # hours: Exophase refreshes a game's counts "hours apart"
REFRESH_GROWTH = 4.0    # players-tracked multiplier at a launch-night refresh
                        # (kit documents 638 -> 3,868, ~6x, at the first one)
JOINER_BETA = 0.10      # fresh joiner's hold prob., as a fraction of the
                        # incumbent rate, for a trophy almost nobody holds
MAX_LOGIT_MOVE = 2.5    # clamp on a trend forecast's move from the last rate
SHRINK = 0.30           # shrinkage of a trophy's move toward the median move
DAMP = 0.90             # damping of a trend extrapolated past the data range


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, residual_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2)) if n > 2 else 0.0
    return a, b, sd


def distinct_states(case):
    """Collapse consecutive identical reads to one state each.

    Returns a list of (hours_since_first_read, players, [holders per trophy]).
    """
    hist = case["history"]
    t0 = parse_time(hist[0]["at"])
    states = []
    prev_key = None
    for read in hist:
        rates = read["rates"]
        key = (read["players_tracked"], tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        hours = (parse_time(read["at"]) - t0).total_seconds() / 3600.0
        holders = [
            (round(r * read["players_tracked"] / 100.0) if r is not None else None)
            for r in rates
        ]
        states.append((hours, read["players_tracked"], holders))
    return states, t0


def quantiles_from_logit(centre_logit, spread):
    qs = [round4(clamp_pct(inv_logit(centre_logit + z * spread))) for z in Z]
    # Enforce non-decreasing after rounding.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_trend(case, states, t0):
    """Trend mode: power-law fits of the rate on log time.

    Per trophy, log(rate) is fitted by least squares against
    ln(1 + hours since the first read) -- the shape a launch's trophy
    rates follow -- and read at the target. Fitting the rate directly
    (rather than holders and players separately) avoids compounding two
    fits' errors: holders / players is the same line in log space.
    Two fits are blended: all distinct states, and the last 24 hours
    only when it holds >= 3 distinct states (the recent slope is what
    a saturating curve is doing now; the house curve baseline fits the
    last 24 hours the same way). The implied logit move from the last
    rate is shrunk 30% toward the cross-trophy median move and damped
    10% for extrapolating past the observed range, then clamped.
    """
    n_trophies = len(case["trophies"])
    target_h = (parse_time(case["target"]) - t0).total_seconds() / 3600.0
    horizon_days = case["horizon_minutes"] / 1440.0

    xs = [math.log1p(h) for h, _, _ in states]
    x_t = math.log1p(target_h)
    last_h = states[-1][0]
    recent_x = math.log1p(max(0.0, last_h - 24.0))
    use_recent = last_h > 24.0 and sum(1 for h in xs if h >= recent_x) >= 3

    last_rates = []
    for i in range(n_trophies):
        h_last = states[-1][2][i]
        last_rates.append(100.0 * h_last / states[-1][1] if h_last is not None else None)

    medians = []
    deltas = []
    resid_sds = []
    for i in range(n_trophies):
        # Log-rate series over distinct states. Reads below 0.05% carry
        # almost no information (fewer than ~1 holder at launch-night
        # player counts) and their log would anchor the slope, so they
        # are left out of the fit.
        pts = []
        for j, (h, n, holders) in enumerate(states):
            if holders[i] is None:
                continue
            r = 100.0 * holders[i] / n
            if r < 0.05:
                continue
            pts.append((xs[j], math.log(r)))
        if len(pts) < 2:
            medians.append((last_rates[i] or 1.0, 0.0))
            deltas.append(0.0)
            resid_sds.append(0.0)
            continue
        px = [p[0] for p in pts]
        py = [p[1] for p in pts]
        a_all, b_all, sd_all = ols(px, py)
        log_rate_target = a_all + max(-1.5, min(1.5, b_all)) * x_t
        if use_recent:
            rx = [p[0] for p in pts if p[0] >= recent_x]
            ry = [p[1] for p in pts if p[0] >= recent_x]
            if len(rx) >= 3:
                a_rec, b_rec, _ = ols(rx, ry)
                log_rate_recent = a_rec + max(-1.5, min(1.5, b_rec)) * x_t
                # The recent slope is what a saturating curve is doing
                # now; the all-history fit stays as a 15% anchor against
                # a noisy recent window.
                log_rate_target = 0.15 * log_rate_target + 0.85 * log_rate_recent
        resid_sds.append(sd_all)
        rate_target = math.exp(log_rate_target)
        last_rate = last_rates[i] if last_rates[i] is not None else rate_target
        delta = logit(rate_target) - logit(last_rate)
        medians.append((last_rate, delta))
        deltas.append(delta)

    median_delta = median(deltas)
    # Spread: the fits' pooled residual (which already contains whatever
    # refresh jumps the trend did not explain) plus the random-walk term.
    # NOTE: raw jumps between distinct states must NOT be added on top:
    # they are mostly the trend itself, and adding sqrt(n_refreshes) of
    # them double-counts it and blows the band up (found by backtest).
    pooled_sd = median(resid_sds) if resid_sds else 0.0
    spread = math.sqrt(
        max(0.15, pooled_sd) ** 2 + (SIGMA_RW * math.sqrt(horizon_days)) ** 2
    )
    spread = max(0.40, min(2.20, spread))

    rows = []
    for trophy, (last_rate, delta) in zip(case["trophies"], medians):
        adj = DAMP * (median_delta + (1.0 - SHRINK) * (delta - median_delta))
        adj = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, adj))
        centre = logit(last_rate) + adj
        rows.append({"id": trophy["id"], "q": quantiles_from_logit(centre, spread)})
    return {"trophies": rows}


def forecast_refresh_prior(case, states, t0):
    """Refresh-prior mode: the page is flat, but the next refresh dilutes.

    Quantiles are a mixture over the refresh landing before the target:
    the 5% quantile is the diluted rate, the median the probability
    weighted centre, the 95% just above the last rate.
    """
    last = case["history"][-1]
    last_rates = last["rates"]
    elapsed_h = (parse_time(case["at"]) - t0).total_seconds() / 3600.0
    horizon_h = case["horizon_minutes"] / 60.0
    p_refresh = min(0.80, (elapsed_h + horizon_h) / REFRESH_SCALE_H)

    rows = []
    for trophy, rate in zip(case["trophies"], last_rates):
        r = rate if rate is not None else 0.0
        beta = JOINER_BETA + (1.0 - JOINER_BETA) * (r / 100.0) ** 2
        r_dil = r * (1.0 + beta * (REFRESH_GROWTH - 1.0)) / REFRESH_GROWTH
        lg = logit(r)
        lg_dil = logit(r_dil)
        centre = lg + p_refresh * (lg_dil - lg)
        qs = [
            round4(clamp_pct(inv_logit(lg_dil - 0.35))),
            round4(clamp_pct(inv_logit(lg_dil + 0.10))),
            round4(clamp_pct(inv_logit(centre))),
            round4(clamp_pct(inv_logit(lg - 0.10))),
            round4(clamp_pct(inv_logit(lg + 0.25))),
        ]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        # No reads at all: a neutral wide forecast is the only honest one.
        rows = [
            {"id": t["id"], "q": [0.05, 0.3, 1.0, 3.0, 9.0]}
            for t in case["trophies"]
        ]
        return {"trophies": rows}
    states, t0 = distinct_states(case)
    moved = len(states) >= 3
    if moved:
        return forecast_trend(case, states, t0)
    return forecast_refresh_prior(case, states, t0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
