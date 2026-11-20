# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/1372755394-ai/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/1372755394-ai/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the
distribution of its Exophase earn rate (percent of tracked players) at the
case's target, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95). One
program, `main.ts` (Node 22, standard library only, no network, deterministic),
run once per case.

Built for the [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) competition **gta6-trophy-rarity-forecast-01**.

make.fun referral code: `qajetvf4`

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

The grader runs it as `node main.ts` with `/input` and `/output` (the
environment variables override both). It reads every `<id>.in` in the input
directory and writes one `<id>.out` per case:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per
trophy, five finite numbers 0–100, non-decreasing, at most 65,536 bytes.

## The model

Exophase does not update its counts continuously: the page is a snapshot that
only changes when Exophase refreshes a game's counts, hours apart. A case's
history is therefore a sequence of flat plateaus ("states") separated by
refresh jumps. A refresh does two things at once: the tracked players keep
playing (rates drift up as they progress) and a batch of new, mostly
just-started players joins the tracked population (rates are diluted down,
hardest for the late, rare trophies). The model treats the history as its
refresh states, not its individual reads, and has two regimes.

**Refresh-mixture regime (fewer than 3 distinct states — the practice cases'
flat launch night).** No trend can be estimated, so the forecast is a
two-branch mixture over "no refresh before the target" (rates stay where they
are) and "a refresh lands" (rates are diluted by the new joiners). The refresh
probability comes from a renewal argument: the page refreshes every few hours
and the current state has already been unchanged for a known age, averaged
over plausible refresh periods (4–12 h). The refresh branch's per-trophy drop
comes from a cohort-mixing model: after a refresh,
rate = (old earners + new joiners who already earned it) / new tracked total,
where a new joiner's chance of holding a trophy is 1-(1-r)^kappa, kappa
growing with the trophy's own rate (early, easy trophies are picked up fast by
new players; late ones are not). Its two constants are the launch prior
documented in the kit (`competition.yaml`, practice section: the first refresh
took the tracked total 638 → 3,868 players and trophies earned 943 → 4,276):
the players-growth factor g ≈ 6.06 and the kappa scale that reproduces that
aggregate. This prior fires only while the history shows no refresh; once
refreshes are observed the trend regime takes over. If exactly one refresh is
observed (2 states), its per-trophy logit drop is blended 50/50 with the
prior drop.

**Trend regime (3+ distinct states after the first population break — the
final's regime).** Per trophy, two predictors of the rate at the target,
averaged in logit space:
- (a) a recency-weighted least-squares line of logit(rate) on ln(1 + hours
  since release), slope shrunk toward its tier's mean slope;
- (b) a saturating penetration curve r(t) = R (1 − exp(−t/τ)), τ by grid
  search: a trophy's rate cannot rise past the share of players who ever
  reach it, so the line's unbounded rise is the wrong shape for a week-one
  forecast.

States before the first population break (one refresh at least doubling the
tracked total — launch night's page was the reviewers' rates, not the
public's) are dropped from the fit. The spread is calibrated on the history
itself: one-step-ahead walk-forward residuals over the states, grown with the
expected number of refreshes in the horizon, plus the slope's own uncertainty
and one refresh-interval of staleness (the answer is the rate at the last
refresh before the target, which lags it). A trophy essentially everyone holds
(≥ 98%) is held where it is with a narrow spread instead of extrapolating
logit noise. An earner/player decomposition (extrapolating log(earners) and
log(players) separately) was tried and dropped: the two linear extrapolations
diverge badly over a multi-day horizon.

## Backtests

The kit publishes no per-trophy practice answers, only the aggregates of the
answering read (players 638 → 3,868, trophies earned 943 → 4,276) and the
house baseline's practice figures (pinball loss ≈ 1.51, typical miss
≈ 0.68 pp, 90% coverage ≈ 0). So the practice answers were reconstructed
from those documented aggregates under two one-parameter cohort-mixing
families, and every candidate was scored against both reconstructions:

- **R1** — new joiners' earn chance 1−(1−r)^κ with one κ for all trophies,
  κ solved from the aggregate (κ = 0.0665). The baseline scores 1.401 on it
  (typical miss 0.677 pp, coverage 0).
- **R2** — κ growing with the trophy's own rate (κ ∝ r²), calibrated to
  reproduce *both* published figures (loss 1.510, miss 0.677 pp). The
  heterogeneous drop (rare trophies fall further in logit terms) is the
  shape the published loss implies.

Pinball loss on the logit scale (lower is better), mean over both practice
cases:

| forecast | R1 | R2 |
|---|---:|---:|
| house persistence baseline | 1.401 | 1.510 |
| **this entry** | **0.302** | **0.321** |

Per case: p00 0.321 / 0.341, p01 0.283 / 0.301 (R1 / R2). Typical miss
0.13 pp (baseline 0.68 pp), 90% coverage 0.982 (baseline 0).

Two honesty checks, reported as they are:

- **Truncated-history backtest (no-refresh regime).** Cutting each practice
  history at an earlier read and forecasting a later read in the same flat
  history (no refresh inside the window): this entry 0.174 vs baseline
  0.025 over 5 truncations. The refresh bet costs ~0.15 when no refresh
  arrives within 90 minutes — the price of the ~4.7× win when one does.
  In expectation at a ~60% refresh probability the entry is ~0.25 vs the
  baseline's ~0.88.
- **Synthetic final-like cases.** A 120-read case was generated with 12
  refresh states, a saturating player inflow (only +22% from checkpoint to
  target) and rising rates, under two different truth shapes, to exercise
  the trend regime end to end (the practice cases never reach it):
  - saturating truth: this entry **0.199**, persistence 0.407, house curve
    baseline 0.308;
  - logit-linear (non-saturating) truth: this entry **0.188**, persistence
    0.449, house curve baseline 0.278.

  The synthetic truths are the author's own constructions, so they test the
  mechanics and the robustness of the blend across truth shapes, not the
  real final's rates.

## Verification

- Both practice outputs valid: 55 rows, one per trophy, each id once, five
  finite numbers 0–100, non-decreasing; 2,777 and 2,768 bytes (limit 65,536).
- Deterministic: two runs write byte-identical outputs.
- Fast: ~0.25 s per case locally (limit 8 s).
- Edge cases tested: 2-state history (mixture with the observed refresh),
  empty history, null rates in reads, single-trophy case — all produce valid
  output.
- `makefun grade-local` reports this competition is graded on the server
  only (scorer and reference data are not in the kit), so there is no local
  grade; the backtests above are the substitute.
- `makefun check --worktree` passes the competition-open, snapshot, license
  (MIT, SPDX) and entrypoint checks; the only failure is
  `transcripts_present`, because the transcripts are bundled at submission
  time, outside this repository.

## Files

- `main.ts` — the entry (Node 22 stdlib only).
- `LICENSE` — MIT (from the kit's starter).
- `.makefun/owner.json` — the entry wallet.
