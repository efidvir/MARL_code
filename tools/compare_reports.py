#!/usr/bin/env python3
"""Compare two confirmatory_analysis.py reports (JSON) leaf by leaf.

Validation of the updated analysis script (RERUN_SPEC WP9): run on the frozen
pickles it must reproduce ~/MARL_run_6/confirmatory/cf_report_A1.json and
dev_report_A1.json for every P1-P9 and S1-S7 value.  This tool walks the
listed sections of REF and NEW and reports every leaf that differs.

Usage:
  compare_reports.py REF.json NEW.json
        [--sections seeds,excluded,descriptive,primary,secondary]
        [--rtol 0] [--atol 0] [--ignore key1,key2]

Numbers compare with math.isclose(rel_tol=rtol, abs_tol=atol) (default: exact;
NaN equals NaN); booleans, strings and null must be equal.  A key REF has and
NEW lacks is a MISMATCH; a key only NEW has is a NOTE (new output, e.g. the
WP9 sections), never a failure.  Exit 0 when no leaf of the listed sections
differs, 1 otherwise, 2 on usage errors.
"""
import argparse
import json
import math
import os
import sys

DEFAULT_SECTIONS = ('seeds', 'excluded', 'descriptive', 'primary', 'secondary')


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _leaf_equal(a, b, rtol, atol):
    if _is_num(a) and _is_num(b):
        if isinstance(a, float) and math.isnan(a) or isinstance(b, float) and math.isnan(b):
            return (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b))
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b and type(a) is type(b)


def compare(ref, new, rtol=0.0, atol=0.0, ignore=(), path=''):
    """Returns (mismatches, notes): mismatches = [(path, ref, new)],
    notes = [path] for keys present only in NEW."""
    mism, notes = [], []
    if isinstance(ref, dict) and isinstance(new, dict):
        for k in ref:
            if k in ignore:
                continue
            p = '%s/%s' % (path, k)
            if k not in new:
                mism.append((p, ref[k], '<missing>'))
                continue
            m, n = compare(ref[k], new[k], rtol, atol, ignore, p)
            mism += m
            notes += n
        for k in new:
            if k not in ref and k not in ignore:
                notes.append('%s/%s' % (path, k))
        return mism, notes
    if isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            mism.append((path, 'len %d' % len(ref), 'len %d' % len(new)))
            return mism, notes
        for i, (x, y) in enumerate(zip(ref, new)):
            m, n = compare(x, y, rtol, atol, ignore, '%s[%d]' % (path, i))
            mism += m
            notes += n
        return mism, notes
    if not _leaf_equal(ref, new, rtol, atol):
        mism.append((path, ref, new))
    return mism, notes


def count_leaves(x):
    if isinstance(x, dict):
        return sum(count_leaves(v) for v in x.values())
    if isinstance(x, list):
        return sum(count_leaves(v) for v in x)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('ref')
    ap.add_argument('new')
    ap.add_argument('--sections', default=','.join(DEFAULT_SECTIONS))
    ap.add_argument('--rtol', type=float, default=0.0)
    ap.add_argument('--atol', type=float, default=0.0)
    ap.add_argument('--ignore', default='', help='comma list of key names skipped at any depth')
    a = ap.parse_args(argv)
    for p in (a.ref, a.new):
        if not os.path.isfile(p):
            print('compare_reports: no such file: %s' % p, file=sys.stderr)
            return 2
    ref, new = json.load(open(a.ref)), json.load(open(a.new))
    ignore = tuple(x for x in a.ignore.split(',') if x)
    total_m, total_leaves = 0, 0
    print('REF %s\nNEW %s' % (a.ref, a.new))
    for sec in [s for s in a.sections.split(',') if s]:
        if sec not in ref:
            print('section %-12s not in REF -- skipped' % sec)
            continue
        if sec not in new:
            print('section %-12s MISSING in NEW' % sec)
            total_m += 1
            continue
        m, n = compare(ref[sec], new[sec], a.rtol, a.atol, ignore, sec)
        leaves = count_leaves(ref[sec])
        total_leaves += leaves
        total_m += len(m)
        print('section %-12s leaves=%-5d mismatches=%-4d new-keys=%d' % (sec, leaves, len(m), len(n)))
        for p, x, y in m[:60]:
            print('   MISMATCH %s: ref=%r new=%r' % (p, x, y))
        if len(m) > 60:
            print('   ... %d more' % (len(m) - 60))
    extra = [k for k in new if k not in ref]
    if extra:
        print('NOTE top-level keys only in NEW: %s' % ', '.join(extra))
    print('SUMMARY leaves=%d mismatches=%d -> %s' % (
        total_leaves, total_m, 'IDENTICAL' if total_m == 0 else 'DIFFERENT'))
    return 0 if total_m == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
