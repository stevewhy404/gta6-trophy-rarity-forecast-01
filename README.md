# gta6-trophy-rarity-forecast-01

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5/25/50/75/95%) of its Exophase earn rate at the case's target,
from the house's recorded reads of the game's trophy page. Scored by
pinball loss on the logit scale; the entry is `main.py` (Python 3.13,
standard library only, offline, deterministic).

## Model

A trophy's earn rate is holders / players-tracked, and Exophase refreshes
the page's counts on its own schedule (hours apart), so displayed rates sit
flat between refreshes and jump at them.

1. **Distinct states.** Consecutive identical reads are one refresh, so the
   history is collapsed to its distinct states.
2. **Trend mode** (history shows real movement). Per trophy, log(rate) is
   fitted by least squares against ln(1 + hours since the first read) and
   read at the target. The fit that matters is the last 24 hours' (the
   recent slope is what a saturating curve is doing now), with the
   all-history fit kept as a 15% anchor; slopes are clamped to ±1.5 and
   reads below 0.05% are left out of the fit. The implied logit move is
   shrunk 30% toward the cross-trophy median move, damped 10%, and
   clamped to 2.5 logit units. The spread combines the fits' pooled
   residual with a random-walk term (0.5 logit per root-day).
3. **Refresh-prior mode** (history is flat). No trend can be fitted, but a
   flat page does not mean the rates stay flat: the next refresh brings
   the players who have synced since, and fresh joiners dilute every rate
   but the universal ones (the kit documents the first refresh taking
   players 638 → 3,868 while every rate but the 100% one fell). The
   forecast is a mixture over the refresh landing before the target:
   the 5% quantile is the diluted rate, the median the
   probability-weighted centre, the 95% just above the last rate.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` per case.
Two runs on the same case write identical bytes. Runtime ~0.1 s per case.

## Validation (local)

The server holds the true practice answers, so the dev grade below is
scored against answers reconstructed from the facts `competition.yaml`
states openly (04:30Z read: 3,868 players, 4,276 trophies earned; every
rate but the 100% one fell; 05:00Z read unchanged), under three ways of
splitting the 103 extra holders across trophies. The reconstruction is
validated by reproducing the house baseline's stated figures (typical miss
0.68 points, coverage 0).

| answer split | this entry | house baseline |
|---|---|---|
| proportional to holders | **0.459** | 1.403 |
| uniform | **0.450** | 1.388 |
| concentrated on big trophies | **0.532** | 1.467 |

Kit states the baseline's practice loss as 1.51. The entry beats it by
~0.9 logit under every reconstruction (typical miss 0.17 vs 0.68 points).

The trend mode (which is what runs on the final, with its 2-day history)
was also backtested on two synthetic final-like cases (48 h of reads,
4-hour refreshes, saturating trophy curves, checkpoint + 4.5-day target):
loss 0.335 vs baseline 0.397 (smooth) and 0.344 vs 0.352 (noisy, faster
growth). Synthetic truth is invented, so these numbers are a machinery
check, not a promise about the real final.
