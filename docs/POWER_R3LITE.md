# The power statement of R3-lite: script and output

This file holds the script that produced the numbers of
`docs/PREREGISTRATION_R3.md` §5.1, and its output, so that the power
statement can be reproduced from the tree. It was added after the review of
2026-10-11 (`docs/DECISIONS.md` D-R3.2) and extended the same day for the
design of D-R3.3: a block of **forty** instances per scenario, and **two
registered arms** — replicate 1234 (family 1, hypotheses P1–P9) and the
ensemble of the four R2 replicates (family E, hypotheses E-P1–E-P9). The
first draft's block of twenty instances is printed beside it, because the
comparison of the two is the reason the block was enlarged.

**What it is.** An analysis of the selection data only. It reads three JSON
files that R3 Phase 0 wrote for the development and validation instances
(`~/MARL_r3/r3out/phase0/results/select_replicate_{dev,val,S1}_min.json`,
sha256 `0df0306727209e5c…`, `cd8ddbcd80024eeb…`, `a61ba7f3cb1e5824…`). It
runs no simulator pass, opens no pickle, reads nothing about any
confirmatory block and writes nothing. It is not part of the chain and no
stage calls it.

**How to run it** (on <compute-server>, about ten minutes on one core):

```bash
cd <run tree> && mkdir -p <server-tmp> && \
  awk '/^```python$/{f=1;next} /^```$/{f=0} f' docs/POWER_R3LITE.md > <server-tmp>/power_r3lite.py && \
  sha256sum <server-tmp>/power_r3lite.py && \
  PYTHONHASHSEED=0 OMP_NUM_THREADS=1 nice -n 19 python3 <server-tmp>/power_r3lite.py
```

The extracted script has sha256
`8d0fe85828978ec121da51c01cb6687c6a6b29e70867d4160f818f9af91cc4b2`.
(The n = 20 script of the first draft had sha256 `bf9138c424e76838…`; its
n = 20 rows for replicate 1234 are reproduced digit for digit below.) It
lives in this document, and not as a file under `tools/`, because it is not
part of the chain.

## Script

```python
#!/usr/bin/env python3
"""The power statement of docs/PREREGISTRATION_R3.md section 5.1.

Analysis of the SELECTION DATA only: it reads the per-instance paired
differences that R3 Phase 0 wrote for the development and validation
instances (results/select_replicate_{dev,val,S1}_min.json), simulates
nothing on the simulator, reads no pickle and writes nothing.

  python3 power_r3lite.py [RESULTS_DIR]     (default ~/MARL_r3/r3out/phase0/results)

What it prints, for EACH of the two registered arms (family 1: replicate
1234; family E: the ensemble of the four replicates; D-R3.3)
  1. the paired differences of the arm against XL-DET: mean, SD,
     standardised effect, median, range (development n = 20, validation n = 7)
     and the correlation of the Scenario A and Scenario B differences;
  2. for n = 40 (the registered block) and for n = 20 (the first draft's
     block, for comparison): the probability that the exact two-sided
     Wilcoxon signed-rank test on n differences gives p below 0.05/9
     (sufficient under Holm over nine) and below 0.05 (necessary) in the
     expected direction, differences drawn from a normal distribution with
     the stated mean and SD; and the probability that the hypothesis against
     XL-DET in Scenario A AND the one in Scenario B both reach p < 0.05/9,
     differences drawn from a bivariate normal with the development
     correlation;
  3. the smallest true effect detected with probability 0.8 and 0.9 at n = 40;
and once
  4. a bootstrap of the SELECTION: instances resampled with replacement, the
     registered rule (largest min over A and B of the mean paired
     difference) re-applied to the five candidates, the pick scored on the
     instances left out -- the optimism of the in-sample estimate.

Every random draw comes from numpy's default generator started at GEN.
"""
import json
import os
import statistics as st
import sys

import numpy as np

GEN = 20261011          # generator start of every simulation below
DRAWS = 20000
N = 40                  # instances per scenario of the confirmatory block (D-R3.3)
N_FIRST_DRAFT = 20      # the first draft's block, printed for comparison
PICKS = (('ck1234', 'family 1, replicate 1234', 'P1', 'P4'),
         ('ens', 'family E, the ensemble', 'E-P1', 'E-P4'))
ALPHA = 0.05
FAMILY = 9
PICK = 'ck1234'
# R2's confirmatory paired SD of P1 and P4 (the R2 report's own figures,
# typed here as constants: no R2 confirmatory data is read)
R2_SD_A, R2_SD_B = 8.397, 10.037


def exact_p(d):
    """(two-sided exact Wilcoxon signed-rank p, W+ > W-) for differences
    without zeros or ties (draws from a continuous distribution)."""
    d = np.asarray(d, float)
    n = len(d)
    order = np.argsort(np.abs(d), kind='stable')
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    dbl = np.rint(2 * ranks).astype(int)
    tot = int(dbl.sum())
    dist = np.zeros(tot + 1)
    dist[0] = 1.0
    for r in dbl:
        dist[r:] = dist[r:] + dist[:tot + 1 - r]
    centre = tot / 2
    wp = ranks[d > 0].sum()
    dev = abs(2 * wp - centre)
    idx = np.arange(tot + 1)
    p = min(1.0, float(dist[np.abs(idx - centre) >= dev - 1e-9].sum() / 2.0 ** n))
    return p, wp > n * (n + 1) / 4


def power(mu, sd, draws=DRAWS, n=N):
    rng = np.random.default_rng(GEN)
    strict = loose = 0
    for _ in range(draws):
        p, pos = exact_p(rng.normal(mu, sd, n))
        strict += int(pos and p < ALPHA / FAMILY)
        loose += int(pos and p < ALPHA)
    return strict / draws, loose / draws


def joint(mu_a, sd_a, mu_b, sd_b, rho, draws=DRAWS, n=N):
    rng = np.random.default_rng(GEN)
    cov = [[sd_a ** 2, rho * sd_a * sd_b], [rho * sd_a * sd_b, sd_b ** 2]]
    both = 0
    for _ in range(draws):
        x = rng.multivariate_normal([mu_a, mu_b], cov, n)
        p1, s1 = exact_p(x[:, 0])
        p4, s4 = exact_p(x[:, 1])
        both += int(s1 and s4 and p1 < ALPHA / FAMILY and p4 < ALPHA / FAMILY)
    return both / draws


def smallest_effect(sd, target, n=N):
    lo, hi = 0.0, 3 * sd
    for _ in range(14):
        mid = (lo + hi) / 2
        if power(mid, sd, draws=4000, n=n)[0] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def selection_optimism(scores, resamples=5000):
    """Bootstrap of the selection on the 27 development + validation
    instances: the registered rule re-applied to each resample, the pick
    scored in the resample and on the instances left out of it."""
    cands = sorted(scores)
    seeds = sorted(scores[PICK]['A']['per_seed'], key=int)
    a = {c: np.array([scores[c]['A']['per_seed'][s] for s in seeds]) for c in cands}
    b = {c: np.array([scores[c]['B']['per_seed'][s] for s in seeds]) for c in cands}
    rng = np.random.default_rng(GEN)
    picks = {c: 0 for c in cands}
    opt_a, opt_b, used = [], [], 0
    for _ in range(resamples):
        idx = rng.integers(0, len(seeds), len(seeds))
        out = np.setdiff1d(np.arange(len(seeds)), idx)
        if len(out) == 0:
            continue
        used += 1
        best = max(cands, key=lambda c: min(a[c][idx].mean(), b[c][idx].mean()))
        picks[best] += 1
        opt_a.append(a[best][idx].mean() - a[best][out].mean())
        opt_b.append(b[best][idx].mean() - b[best][out].mean())
    return (len(seeds), used, {c: picks[c] / used for c in cands},
            float(np.mean(opt_a)), float(np.mean(opt_b)),
            float(np.percentile(opt_a, 97.5)), float(np.percentile(opt_b, 97.5)))


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/MARL_r3/r3out/phase0/results')
    load = lambda name: json.load(open(os.path.join(res, name)))['scores']
    dev, val, both = (load('select_replicate_%s_min.json' % s) for s in ('dev', 'val', 'S1'))
    for pick, label, h_a, h_b in PICKS:
        print('==== %s (%s), paired differences against XL-DET in percentage points' % (label, pick))
        stat = {}
        for name, o in (('development', dev[pick]), ('validation', val[pick])):
            for k in 'AB':
                d = list(o[k]['per_seed'].values())
                stat[name, k] = (st.mean(d), st.stdev(d))
                print('1. %-11s %s  n %2d  mean %+.2f  SD %.2f  d_z %.2f  median %+.2f  range %+.2f to %+.2f'
                      % (name, k, len(d), st.mean(d), st.stdev(d), st.mean(d) / st.stdev(d),
                         st.median(d), min(d), max(d)))
        ds = sorted(dev[pick]['A']['per_seed'], key=int)
        da = np.array([dev[pick]['A']['per_seed'][s] for s in ds])
        db = np.array([dev[pick]['B']['per_seed'][s] for s in ds])
        rho = float(np.corrcoef(da, db)[0, 1])
        print('1. correlation of the A and B differences across the %d development instances: %.2f'
              % (len(ds), rho))
        (ma, sa), (mb, sb) = stat['development', 'A'], stat['development', 'B']
        (va, _), (vb, _) = stat['validation', 'A'], stat['validation', 'B']
        rows = (('the development estimate', ma, sa, mb, sb, 1.0),
                ('half of it', ma / 2, sa, mb / 2, sb, 0.5),
                ('a third of it', ma / 3, sa, mb / 3, sb, 1 / 3.0),
                ('the validation estimate, development SD', va, sa, vb, sb, None),
                ('the development estimate, R2 confirmatory SD', ma, R2_SD_A, mb, R2_SD_B, None))
        for n in (N, N_FIRST_DRAFT):
            print('2. n = %d: probability of p < 0.05/9 | p < 0.05, expected direction, %d draws; '
                  'last column: %s AND %s both at p < 0.05/9 (correlation %.2f)' % (n, DRAWS, h_a, h_b, rho))
            for name, a_mu, a_sd, b_mu, b_sd, f in rows:
                pa, pb = power(a_mu, a_sd, n=n), power(b_mu, b_sd, n=n)
                both_p = '%.3f' % joint(a_mu, a_sd, b_mu, b_sd, rho, n=n) if a_sd == sa else '    -'
                print('   %-45s %s (A) mean %+6.2f SD %5.2f: %.3f | %.3f    %s (B) mean %+6.2f SD %5.2f: %.3f | %.3f    both %s'
                      % (name, h_a, a_mu, a_sd, pa[0], pa[1], h_b, b_mu, b_sd, pb[0], pb[1], both_p))
        print('3. n = %d: smallest true effect detected at p < 0.05/9 with probability 0.8 / 0.9' % N)
        for k, sd in (('A', sa), ('B', sb)):
            e8, e9 = smallest_effect(sd, 0.8), smallest_effect(sd, 0.9)
            print('   %s  %.2f / %.2f pp  (d_z %.2f / %.2f)' % (k, e8, e9, e8 / sd, e9 / sd))
    n, used, picks, oa, ob, oa97, ob97 = selection_optimism(both)
    print('==== 4. bootstrap of the selection (%d instances, %d resamples with instances left out)' % (n, used))
    print('   picked: %s' % ', '.join('%s %.3f' % (c, picks[c]) for c in sorted(picks)))
    print('   optimism of the pick (in the resample minus left out): A %+.2f pp (97.5th percentile %+.2f), '
          'B %+.2f pp (%+.2f)' % (oa, oa97, ob, ob97))


if __name__ == '__main__':
    main()
```

## Output (<compute-server>, 2026-10-11)

```
==== family 1, replicate 1234 (ck1234), paired differences against XL-DET in percentage points
1. development A  n 20  mean +10.11  SD 9.06  d_z 1.12  median +9.90  range -7.20 to +29.16
1. development B  n 20  mean +12.86  SD 6.76  d_z 1.90  median +10.71  range -0.33 to +25.11
1. validation  A  n  7  mean +5.39  SD 6.47  d_z 0.83  median +8.31  range -6.62 to +12.14
1. validation  B  n  7  mean +15.19  SD 11.10  d_z 1.37  median +18.06  range -4.75 to +26.86
1. correlation of the A and B differences across the 20 development instances: 0.33
2. n = 40: probability of p < 0.05/9 | p < 0.05, expected direction, 20000 draws; last column: P1 AND P4 both at p < 0.05/9 (correlation 0.33)
   the development estimate                      P1 (A) mean +10.11 SD  9.06: 1.000 | 1.000    P4 (B) mean +12.86 SD  6.76: 1.000 | 1.000    both 1.000
   half of it                                    P1 (A) mean  +5.06 SD  9.06: 0.692 | 0.918    P4 (B) mean  +6.43 SD  6.76: 0.997 | 1.000    both 0.695
   a third of it                                 P1 (A) mean  +3.37 SD  9.06: 0.277 | 0.610    P4 (B) mean  +4.29 SD  6.76: 0.828 | 0.968    both 0.249
   the validation estimate, development SD       P1 (A) mean  +5.39 SD  9.06: 0.766 | 0.946    P4 (B) mean +15.19 SD  6.76: 1.000 | 1.000    both 0.764
   the development estimate, R2 confirmatory SD  P1 (A) mean +10.11 SD  8.40: 1.000 | 1.000    P4 (B) mean +12.86 SD 10.04: 1.000 | 1.000    both     -
2. n = 20: probability of p < 0.05/9 | p < 0.05, expected direction, 20000 draws; last column: P1 AND P4 both at p < 0.05/9 (correlation 0.33)
   the development estimate                      P1 (A) mean +10.11 SD  9.06: 0.942 | 0.996    P4 (B) mean +12.86 SD  6.76: 1.000 | 1.000    both 0.940
   half of it                                    P1 (A) mean  +5.06 SD  9.06: 0.269 | 0.634    P4 (B) mean  +6.43 SD  6.76: 0.824 | 0.975    both 0.246
   a third of it                                 P1 (A) mean  +3.37 SD  9.06: 0.089 | 0.331    P4 (B) mean  +4.29 SD  6.76: 0.373 | 0.746    both 0.053
   the validation estimate, development SD       P1 (A) mean  +5.39 SD  9.06: 0.318 | 0.690    P4 (B) mean +15.19 SD  6.76: 1.000 | 1.000    both 0.322
   the development estimate, R2 confirmatory SD  P1 (A) mean +10.11 SD  8.40: 0.971 | 0.999    P4 (B) mean +12.86 SD 10.04: 0.986 | 0.999    both     -
3. n = 40: smallest true effect detected at p < 0.05/9 with probability 0.8 / 0.9
   A  5.65 / 6.33 pp  (d_z 0.62 / 0.70)
   B  4.21 / 4.72 pp  (d_z 0.62 / 0.70)
==== family E, the ensemble (ens), paired differences against XL-DET in percentage points
1. development A  n 20  mean +6.26  SD 8.60  d_z 0.73  median +6.16  range -15.28 to +23.70
1. development B  n 20  mean +9.16  SD 8.92  d_z 1.03  median +9.46  range -9.50 to +29.54
1. validation  A  n  7  mean +3.84  SD 2.75  d_z 1.39  median +3.00  range +0.05 to +7.54
1. validation  B  n  7  mean +11.35  SD 9.06  d_z 1.25  median +11.48  range +2.36 to +24.10
1. correlation of the A and B differences across the 20 development instances: 0.50
2. n = 40: probability of p < 0.05/9 | p < 0.05, expected direction, 20000 draws; last column: E-P1 AND E-P4 both at p < 0.05/9 (correlation 0.50)
   the development estimate                      E-P1 (A) mean  +6.26 SD  8.60: 0.932 | 0.991    E-P4 (B) mean  +9.16 SD  8.92: 0.999 | 1.000    both 0.932
   half of it                                    E-P1 (A) mean  +3.13 SD  8.60: 0.264 | 0.591    E-P4 (B) mean  +4.58 SD  8.92: 0.595 | 0.868    both 0.212
   a third of it                                 E-P1 (A) mean  +2.09 SD  8.60: 0.090 | 0.309    E-P4 (B) mean  +3.05 SD  8.92: 0.224 | 0.540    both 0.047
   the validation estimate, development SD       E-P1 (A) mean  +3.84 SD  8.60: 0.441 | 0.766    E-P4 (B) mean +11.35 SD  8.92: 1.000 | 1.000    both 0.439
   the development estimate, R2 confirmatory SD  E-P1 (A) mean  +6.26 SD  8.40: 0.945 | 0.993    E-P4 (B) mean  +9.16 SD 10.04: 0.995 | 1.000    both     -
2. n = 20: probability of p < 0.05/9 | p < 0.05, expected direction, 20000 draws; last column: E-P1 AND E-P4 both at p < 0.05/9 (correlation 0.50)
   the development estimate                      E-P1 (A) mean  +6.26 SD  8.60: 0.520 | 0.851    E-P4 (B) mean  +9.16 SD  8.92: 0.889 | 0.988    both 0.498
   half of it                                    E-P1 (A) mean  +3.13 SD  8.60: 0.085 | 0.320    E-P4 (B) mean  +4.58 SD  8.92: 0.215 | 0.560    both 0.042
   a third of it                                 E-P1 (A) mean  +2.09 SD  8.60: 0.033 | 0.165    E-P4 (B) mean  +3.05 SD  8.92: 0.073 | 0.289    both 0.009
   the validation estimate, development SD       E-P1 (A) mean  +3.84 SD  8.60: 0.148 | 0.452    E-P4 (B) mean +11.35 SD  8.92: 0.984 | 0.999    both 0.150
   the development estimate, R2 confirmatory SD  E-P1 (A) mean  +6.26 SD  8.40: 0.547 | 0.867    E-P4 (B) mean  +9.16 SD 10.04: 0.780 | 0.965    both     -
3. n = 40: smallest true effect detected at p < 0.05/9 with probability 0.8 / 0.9
   A  5.37 / 6.00 pp  (d_z 0.62 / 0.70)
   B  5.57 / 6.23 pp  (d_z 0.62 / 0.70)
==== 4. bootstrap of the selection (27 instances, 5000 resamples with instances left out)
   picked: ck1234 0.994, ck2345 0.000, ck3456 0.000, ck4567 0.000, ens 0.006
   optimism of the pick (in the resample minus left out): A +0.01 pp (97.5th percentile +7.10), B -0.06 pp (+6.37)
```

## Reading

- The n = 40 blocks of section 2 are the tables of §5.1. "p < 0.05/9" is
  sufficient for confirmation under Holm whatever the other eight p-values
  of the family are; "p < 0.05" is necessary. The truth for a hypothesis
  lies between the two columns.
- **Family 1 (replicate 1234).** At the development estimate everything is
  near 1 at either block size. The block size matters when the effect is
  smaller than on the selection data: at the validation-level estimate in
  Scenario A (+5.39 pp) P1 reaches p < 0.05/9 | p < 0.05 with probability
  0.32 | 0.69 on twenty instances and 0.77 | 0.95 on forty; at half the
  development estimate 0.27 | 0.63 and 0.69 | 0.92. P4 (Scenario B) is
  adequately powered at half the estimate on either block.
- **Family E (the ensemble).** Its development effects are smaller (+6.26 and
  +9.16 pp against +10.11 and +12.86). On twenty instances E-P1 would have
  had 0.52 | 0.85 even at the development estimate; on forty it has 0.93 |
  0.99, and E-P4 0.999. **Below the development estimate family E stays
  under-powered on forty instances**: at half of it E-P1 0.26 | 0.59 and
  E-P4 0.60 | 0.87; at the validation-level estimate (+3.84 pp in A) E-P1
  0.44 | 0.77.
- Section 4 answers one narrow question: how much of the development +
  validation estimate is optimism from picking the best of five candidates
  on those instances? Almost none (the registered rule picks replicate 1234
  in 99.4 % of the resamples; the pick's estimate in a resample exceeds its
  estimate on the instances left out by 0.0 pp in Scenario A and −0.1 pp in
  B on average), because replicate 1234 leads the second candidate by more
  than three points in both scenarios.
- It does **not** follow that the estimates will hold. Resampling these
  instances cannot measure what matters more: the development instances
  were used throughout R2's development (R3 §3.3 item 2), and forty new
  topologies need not resemble them. The validation estimate in Scenario A
  is 53 % of the development one for replicate 1234 and 61 % for the
  ensemble. The rows "half" and "a third" are stress levels chosen for that
  reason; they are not derived from the selection spread.
- R3 Phase 0 also ran a different power calculation: a bootstrap of the 20
  development instances under the full Holm family of nine
  (`r3out/phase0/tools/power_r3.py`, sha256 `bd50c718e8eeb969…`, output
  `results/power_r3_bootstrap.json`, sha256 `9791d28681f9875f…`). It
  resamples the selection data at their observed effect and so describes
  the first row of each block only (at n = 20: 0.998 for P1 and for "P1 and
  P4 both"; 0.885 for the ensemble's contrast against XL-DET in A).
