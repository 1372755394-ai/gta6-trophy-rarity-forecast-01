// gta6-trophy-rarity-forecast-01 (Node 22, standard library only).
//
// Forecasts, for every GTA VI trophy, the distribution of its Exophase earn rate
// (percent of tracked players) at the case's target, as five quantiles.
//
// MODEL
// Exophase does not update its counts continuously: the page is a snapshot that
// only changes when Exophase refreshes a game's counts, hours apart. So a case's
// history is a sequence of flat plateaus ("states") separated by refresh jumps.
// A refresh does two things at once: the tracked players keep playing (rates
// drift up as they progress through the game) and a batch of new, mostly
// just-started players joins the tracked population (rates are diluted down,
// hardest for the late, rare trophies). The model therefore treats the history
// as its refresh states, not as its individual reads, and has two regimes:
//
//  1. Trend regime (3+ distinct states after the first population break, the
//     final's regime). Per trophy, two predictors of the rate at the target,
//     averaged in logit space (neither functional form is trusted alone over
//     a multi-day horizon):
//       (a) a recency-weighted least-squares line of logit(rate) on
//           ln(1 + hours since release), slope shrunk toward its tier's mean;
//       (b) a saturating penetration curve r(t) = R (1 - exp(-t / tau)), tau
//           by grid search: a trophy's rate cannot rise past the share of
//           players who ever reach it, so the line's unbounded rise is the
//           wrong shape for a week-one forecast.
//     States before the first population break (one refresh at least doubling
//     the tracked total: launch night's page was the reviewers' rates, not
//     the public's) are dropped from the fit. The spread is calibrated on
//     the history itself: one-step-ahead walk-forward residuals over the
//     states, grown with the expected number of refreshes in the horizon,
//     plus the slope's own uncertainty and one refresh-interval of staleness
//     (the answer is the rate at the last refresh before the target, which
//     lags it). A trophy essentially everyone holds (>= 98%) is held where
//     it is with a narrow spread instead of extrapolating logit noise.
//     (An earner/player decomposition was tried and dropped: extrapolating
//     log(earners) and log(players) separately diverges badly over a
//     multi-day horizon, and the saturating curve already carries the
//     slowing-inflow effect.)
//
//  2. Refresh-mixture regime (fewer than 3 distinct states, e.g. the practice
//     cases' flat launch night). No trend can be estimated, so the forecast is
//     a two-branch mixture over "no refresh before the target" (rates stay
//     where they are) and "a refresh lands" (rates are diluted by the new
//     joiners). The refresh probability comes from a renewal argument: the
//     page refreshes every few hours and the current state has already been
//     unchanged for a known age, averaged over plausible refresh periods.
//     The refresh branch's per-trophy drop comes from a cohort-mixing model:
//     after a refresh, rate = (old earners + new joiners who already earned
//     it) / new tracked total, where a new joiner's chance of holding a trophy
//     is 1-(1-r)^kappa, kappa growing with the trophy's own rate (early, easy
//     trophies are picked up fast by new players; late ones are not). Its two
//     constants are the launch prior documented in the kit
//     (competition.yaml, practice section: the first refresh took the tracked
//     total 638 -> 3,868 players and trophies earned 943 -> 4,276): the
//     players-growth factor g and the kappa scale that reproduces that
//     aggregate. This prior fires only while the history shows no refresh;
//     once refreshes are observed the trend regime takes over.
//
// Everything is deterministic (no randomness, no clock, no network); the same
// case always writes the same bytes. Runs offline, Node 22 stdlib only.
//
// Usage (as the grader runs it): node main.ts, reading $MAKEFUN_INPUT_DIR
// (else /input) and writing $MAKEFUN_OUTPUT_DIR (else /output); locally:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** A trophy's forecast: five quantiles of its earn rate (percent) at the target. */
interface TrophyForecast { id: number; q: number[]; }

interface Trophy { id: number; tier: string; }
interface Read {
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game?: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

/** One distinct page state: the first read that showed it, and its numbers. */
interface State {
  atMs: number;
  hoursSinceRelease: number;
  players: number;
  rates: Array<number | null>;
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const c = clampRate(pct);
  return Math.log(c / (100 - c));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Standard normal CDF (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7). */
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-(x * x) / 2);
  let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  if (x > 0) p = 1 - p;
  return p;
}

/** Standard normal quantile (Acklam's approximation, |error| < 1.2e-9). */
function normInv(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let x: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= pHigh) {
    q = p - 0.5;
    const r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  return x;
}

/** Median of a numeric array (a copy is sorted; empty gives NaN). */
function medianOf(values: number[]): number {
  if (values.length === 0) return NaN;
  const s = values.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Weighted least squares of y on x (with intercept). Returns slope, intercept, and the weighted mean of x. */
function wls(xs: number[], ys: number[], ws: number[]): { a: number; b: number; xBar: number; sxx: number } {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  const xBar = sx / sw;
  const yBar = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - xBar) * (xs[i] - xBar);
    sxy += ws[i] * (xs[i] - xBar) * (ys[i] - yBar);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  return { a: yBar - b * xBar, b, xBar, sxx };
}

// ---------------------------------------------------------------------------
// Refresh-mixture regime (fewer than 3 distinct states)
// ---------------------------------------------------------------------------

/** Plausible Exophase refresh periods (hours), averaged over: the kit documents one interval of at least ~4.5 h. */
const REFRESH_PERIODS_H = [4, 5, 6, 8, 12];
/** Launch prior: players-growth factor at the first refresh (kit: 638 -> 3,868). */
const LAUNCH_G = 3868 / 638;
/** Launch prior: kappa scale of the cohort model (kit aggregate: trophies 943 -> 4,276). */
const LAUNCH_KAPPA0 = 0.0654;
/** Launch prior: kappa grows with the trophy's own rate (percent) to this power. */
const LAUNCH_KAPPA_GAMMA = 2;
/** Spread (logit) of the no-refresh branch, per square-root day (the house baseline's rate). */
const MIX_S0 = 0.5;
/** Spread (logit) of the refresh branch: cohort-model uncertainty. */
const MIX_S1 = 0.5;

/** The cohort model's refresh-branch rate for a trophy last at `r` percent. */
function cohortRefreshRate(r: number): number {
  if (r >= 99.99) return r;
  const p = r / 100;
  const kappa = LAUNCH_KAPPA0 * Math.pow(r, LAUNCH_KAPPA_GAMMA);
  const rho = 1 - Math.pow(1 - p, kappa);
  return ((p + (LAUNCH_G - 1) * rho) / LAUNCH_G) * 100;
}

/**
 * P(at least one refresh in the horizon) by a renewal argument: refreshes come
 * roughly every T hours, the current state has already been unchanged for
 * `stateAgeH` hours, so the wait to the next refresh is roughly uniform on
 * [0, T - stateAgeH]; averaged over plausible T. For long horizons this is ~1.
 */
function refreshProbability(horizonH: number, stateAgeH: number): number {
  let p = 0;
  for (const t of REFRESH_PERIODS_H) {
    const window = Math.max(0.25, t - stateAgeH);
    p += Math.min(1, horizonH / window);
  }
  p /= REFRESH_PERIODS_H.length;
  return Math.min(0.98, Math.max(0.02, p));
}

/** The quantiles of a two-branch normal mixture in logit space, by bisection on its CDF. */
function mixtureQuantiles(mu0: number, s0: number, mu1: number, s1: number, p1: number, levels: number[]): number[] {
  const cdf = (x: number): number =>
    (1 - p1) * normCdf((x - mu0) / s0) + p1 * normCdf((x - mu1) / s1);
  return levels.map((tau) => {
    let lo = Math.min(mu0, mu1) - 8 * Math.max(s0, s1) - 1;
    let hi = Math.max(mu0, mu1) + 8 * Math.max(s0, s1) + 1;
    for (let it = 0; it < 80; it++) {
      const mid = (lo + hi) / 2;
      if (cdf(mid) < tau) lo = mid;
      else hi = mid;
    }
    return invLogit((lo + hi) / 2);
  });
}

/**
 * Mixture-regime forecast for one trophy: branch 0 is the last rate, branch 1
 * the cohort model's refreshed rate. `dropObserved` is the per-trophy logit
 * drop seen at the one refresh in the history, when there is one (2 states):
 * it is blended 50/50 with the prior drop.
 */
function mixtureForecast(
  lastRate: number,
  dropObserved: number | null,
  horizonDays: number,
  horizonH: number,
  stateAgeH: number,
  levels: number[],
): number[] {
  // A long horizon with no trend to estimate: say so with a wide, flat spread.
  if (horizonH > 24) {
    const l = logit(lastRate);
    return levels.map((tau) => invLogit(l + normInv(tau) * 1.0));
  }
  const p = refreshProbability(horizonH, stateAgeH);
  const priorDrop = logit(lastRate) - logit(cohortRefreshRate(lastRate));
  const drop = dropObserved === null ? priorDrop : 0.5 * priorDrop + 0.5 * dropObserved;
  const s0 = MIX_S0 * Math.sqrt(Math.max(horizonDays, 1 / 24));
  return mixtureQuantiles(logit(lastRate), Math.max(s0, 0.05), logit(lastRate) - drop, MIX_S1, p, levels);
}

// ---------------------------------------------------------------------------
// Trend regime (3+ distinct states)
// ---------------------------------------------------------------------------

/** Recency half-life (hours) for the trend fit's weights. */
const TREND_HALF_LIFE_H = 24;
/** Slope shrinkage: a trophy's slope is pulled toward its tier's mean slope with this pseudo-count. */
const SHRINK_N = 4;
/** Spread floor / cap (logit units). */
const SIGMA_MIN = 0.5;
const SIGMA_MAX = 2.0;

function forecastCase(c: Case): TrophyForecast[] {
  // Defensive defaults: a malformed case still gets a valid forecast rather
  // than a crash (a crash scores the case as persistence and forfeits prizes).
  const levels = Array.isArray(c.levels) && c.levels.length > 0 ? c.levels : [0.05, 0.25, 0.5, 0.75, 0.95];
  const horizonMinutes = Number.isFinite(c.horizon_minutes)
    ? c.horizon_minutes
    : Number.isFinite(Date.parse(c.target)) && Number.isFinite(Date.parse(c.at))
      ? (Date.parse(c.target) - Date.parse(c.at)) / 60_000
      : 0;
  const zs = levels.map((t) => normInv(t));
  const horizonDays = Math.max(horizonMinutes, 0) / 1440;
  const horizonH = Math.max(horizonMinutes, 0) / 60;
  const checkpointMs = Date.parse(c.at);
  const targetMs = Number.isFinite(Date.parse(c.target)) ? Date.parse(c.target) : checkpointMs + horizonMinutes * 60_000;
  const releaseMs = c.game?.release_date ? Date.parse(`${c.game.release_date}T00:00:00Z`) : NaN;
  const history: Read[] = Array.isArray(c.history) ? c.history : [];
  const t0Ms = Number.isFinite(releaseMs)
    ? releaseMs
    : history.length > 0
      ? Date.parse(history[0].at)
      : checkpointMs;
  const nT = c.trophies.length;

  // --- collapse the history into distinct page states ---
  const states: State[] = [];
  for (const h of history) {
    const prev = states.length > 0 ? states[states.length - 1] : undefined;
    const same =
      prev !== undefined &&
      prev.players === h.players_tracked &&
      prev.rates.length === h.rates.length &&
      prev.rates.every((r, i) => r === h.rates[i]);
    if (!same) {
      states.push({
        atMs: Date.parse(h.at),
        hoursSinceRelease: Math.max(0, (Date.parse(h.at) - t0Ms) / 3_600_000),
        players: h.players_tracked,
        rates: h.rates.slice(),
      });
    }
  }
  // A population break: one refresh at least doubles the tracked total. The
  // states before it belong to a different population (launch night's page was
  // the reviewers' rates, not the public's) and would corrupt any trend, so
  // they are dropped from the trend fit. The first break is the one that
  // matters; later doublings are ordinary growth of the same population.
  let firstBreak = -1;
  for (let s = 1; s < states.length; s++) {
    if (states[s].players >= 2 * Math.max(1, states[s - 1].players)) {
      firstBreak = s;
      break;
    }
  }
  const trendStates = firstBreak > 0 ? states.slice(firstBreak) : states;
  // Last non-null rate per trophy (a trophy missing from the last read still has a history).
  const lastKnown: number[] = [];
  for (let i = 0; i < nT; i++) {
    let v: number | null = null;
    for (let s = states.length - 1; s >= 0; s--) {
      if (states[s].rates[i] !== null && states[s].rates[i] !== undefined) {
        v = states[s].rates[i] as number;
        break;
      }
    }
    lastKnown.push(v === null ? 1 : v);
  }
  const xTarget = Math.log(1 + Math.max(0, (targetMs - t0Ms) / 3_600_000));

  // --- no history at all: valid but uninformative ---
  if (states.length === 0) {
    return c.trophies.map((t) => ({
      id: t.id,
      q: zs.map((z) => round4(invLogit(logit(1) + z * 1.5))),
    }));
  }

  // --- mixture regime: fewer than 3 distinct states after the population break ---
  if (trendStates.length < 3) {
    // The current state's age is the time since its first read (the last state's first read).
    const stateAgeH = Math.max(0, (checkpointMs - trendStates[trendStates.length - 1].atMs) / 3_600_000);
    // With exactly 2 states there is one observed refresh: its per-trophy logit drop.
    const dropObserved: Array<number | null> = new Array(nT).fill(null);
    if (trendStates.length === 2) {
      for (let i = 0; i < nT; i++) {
        const a = trendStates[0].rates[i];
        const b = trendStates[1].rates[i];
        if (a !== null && b !== null) dropObserved[i] = Math.max(-0.5, logit(a) - logit(b));
      }
    }
    return c.trophies.map((t, i) => {
      const q = mixtureForecast(lastKnown[i], dropObserved[i], horizonDays, horizonH, stateAgeH, levels);
      const rounded = q.map((v) => round4(Math.min(100, Math.max(0, v))));
      for (let j = 1; j < rounded.length; j++) if (rounded[j] < rounded[j - 1]) rounded[j] = rounded[j - 1];
      return { id: t.id, q: rounded };
    });
  }

  // --- trend regime (on the post-break states) ---
  const xs = trendStates.map((s) => Math.log(1 + s.hoursSinceRelease));
  const lastT = trendStates[trendStates.length - 1].hoursSinceRelease;
  const ws = trendStates.map((s) => Math.pow(0.5, (lastT - s.hoursSinceRelease) / TREND_HALF_LIFE_H));
  const tTargetH = Math.max(0, (targetMs - t0Ms) / 3_600_000);

  // First pass: per-trophy slopes of the direct rate trend, and tier means.
  const tierSlopes = new Map<string, number[]>();
  const raw: Array<{ a: number; b: number; n: number; xBar: number; sxx: number }> = [];
  for (let i = 0; i < nT; i++) {
    const px: number[] = [];
    const py: number[] = [];
    const pw: number[] = [];
    for (let s = 0; s < trendStates.length; s++) {
      const r = trendStates[s].rates[i];
      if (r === null || r === undefined) continue;
      px.push(xs[s]);
      py.push(logit(r));
      pw.push(ws[s]);
    }
    if (px.length >= 2) {
      raw.push({ ...wls(px, py, pw), n: px.length });
    } else {
      raw.push({ a: logit(lastKnown[i]), b: 0, n: px.length, xBar: xs[xs.length - 1], sxx: 0 });
    }
    const tier = c.trophies[i].tier;
    if (!tierSlopes.has(tier)) tierSlopes.set(tier, []);
    tierSlopes.get(tier)?.push(raw[i].b);
  }
  const tierMean = new Map<string, number>();
  for (const [tier, bs] of tierSlopes) tierMean.set(tier, bs.reduce((a, b) => a + b, 0) / bs.length);
  const globalMean = raw.reduce((a, f) => a + f.b, 0) / Math.max(1, raw.length);

  // Saturating penetration fit: r(t) = R * (1 - exp(-t / tau)), tau by grid
  // search, R by weighted least squares at each tau. A trophy's rate cannot
  // rise past the share of players who ever reach it, so the log-time line's
  // unbounded rise is the wrong shape for a multi-day horizon.
  const TAU_GRID_H = [3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 350, 600, 1000];
  function saturatingPrediction(i: number): number {
    let best: { sse: number; rPred: number } | null = null;
    for (const tau of TAU_GRID_H) {
      let num = 0;
      let den = 0;
      let sse = 0;
      const fs: number[] = [];
      const rs: number[] = [];
      const wts: number[] = [];
      for (let s = 0; s < trendStates.length; s++) {
        const r = trendStates[s].rates[i];
        if (r === null || r === undefined) continue;
        const f = 1 - Math.exp(-trendStates[s].hoursSinceRelease / tau);
        fs.push(f);
        rs.push(r);
        wts.push(ws[s]);
        num += ws[s] * r * f;
        den += ws[s] * f * f;
      }
      if (den <= 1e-12 || fs.length < 2) continue;
      const rInf = Math.min(99.5, Math.max(0.01, num / den));
      for (let k = 0; k < fs.length; k++) {
        const e = rs[k] - rInf * fs[k];
        sse += wts[k] * e * e;
      }
      const rPred = rInf * (1 - Math.exp(-tTargetH / tau));
      if (best === null || sse < best.sse) best = { sse, rPred };
    }
    return best === null ? lastKnown[i] : Math.min(99.5, Math.max(0.01, best.rPred));
  }

  // Second pass: two predictors per trophy, averaged in logit space.
  const mu: number[] = new Array(nT).fill(0);
  for (let i = 0; i < nT; i++) {
    const f = raw[i];
    const prior = tierMean.get(c.trophies[i].tier) ?? globalMean;
    const b = (f.n * f.b + SHRINK_N * prior) / (f.n + SHRINK_N);
    // Re-centre the intercept on the weighted mean point so the shrunk line still fits the data's centre.
    const yBar = f.a + f.b * f.xBar;
    const a = yBar - b * f.xBar;
    const mu1 = a + b * xTarget;
    const mu3 = logit(saturatingPrediction(i));
    mu[i] = Math.min(logit(99.5), Math.max(logit(0.01), (mu1 + mu3) / 2));
  }

  // Walk-forward residuals: fit the direct rate trend on each prefix of states, predict the next state.
  const residuals: number[] = [];
  for (let k = 3; k <= trendStates.length; k++) {
    const subXs = xs.slice(0, k - 1);
    const subWs = ws.slice(0, k - 1);
    const xNext = xs[k - 1];
    for (let i = 0; i < nT; i++) {
      const rNext = trendStates[k - 1].rates[i];
      if (rNext === null || rNext === undefined) continue;
      const px: number[] = [];
      const py: number[] = [];
      const pw: number[] = [];
      for (let s = 0; s < k - 1; s++) {
        const r = trendStates[s].rates[i];
        if (r === null || r === undefined) continue;
        px.push(subXs[s]);
        py.push(logit(r));
        pw.push(subWs[s]);
      }
      if (px.length < 2) continue;
      const fit = wls(px, py, pw);
      residuals.push(logit(rNext) - (fit.a + fit.b * xNext));
    }
  }
  const absRes = residuals.map((e) => Math.abs(e)).sort((a, b) => a - b);
  const sWf = absRes.length >= 10 ? 1.4826 * medianOf(absRes) : 0.5;

  // Mean gap between states (hours) and the staleness of the last refresh before the target.
  let gapSum = 0;
  for (let s = 1; s < trendStates.length; s++) gapSum += trendStates[s].hoursSinceRelease - trendStates[s - 1].hoursSinceRelease;
  const meanGapH = trendStates.length > 1 ? gapSum / (trendStates.length - 1) : 6;
  const staleDx = Math.log(1 + tTargetH) - Math.log(1 + Math.max(0, tTargetH - meanGapH));
  // Expected refreshes between checkpoint and target (the unpredictable part
  // of each refresh accumulates like a random walk).
  const nRefreshes = Math.min(25, Math.max(1, horizonH / Math.max(1, meanGapH)));

  return c.trophies.map((t, i) => {
    // A trophy essentially everyone holds cannot meaningfully fall: hold it
    // where it is with a narrow spread instead of extrapolating logit noise.
    if (lastKnown[i] >= 98) {
      const l = logit(lastKnown[i]);
      const q = zs.map((z) => round4(invLogit(l + z * 0.3)));
      for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
      return { id: t.id, q: q.map((v) => Math.min(100, Math.max(0, v))) };
    }
    const f = raw[i];
    // Slope uncertainty: residual scale over the spread of x, times the extrapolation distance.
    const se = f.sxx > 1e-12 ? sWf / Math.sqrt(f.sxx) : sWf;
    const slopeTerm = se * Math.max(0, xTarget - f.xBar);
    const staleTerm = Math.abs(f.b) * staleDx;
    const sigma = Math.min(
      SIGMA_MAX,
      Math.max(SIGMA_MIN, Math.sqrt(sWf * sWf * (1 + 0.5 * nRefreshes) + slopeTerm * slopeTerm + staleTerm * staleTerm)),
    );
    const q = zs.map((z) => round4(invLogit(mu[i] + z * sigma)));
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: t.id, q: q.map((v) => Math.min(100, Math.max(0, v))) };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${file.slice(0, -'.in'.length)}.out`), `${JSON.stringify({ trophies })}\n`);
}
