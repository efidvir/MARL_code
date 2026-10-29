#!/usr/bin/env python3
"""Sample the memory of a process tree until its root process exits (Linux).

Usage: rss_monitor.py ROOT_PID OUT.tsv [--interval SECONDS]

Every interval, the root and all its descendants are read from /proc:
  utc  elapsed_s  n_procs  sum_rss_mb  sum_pss_mb  max_rss_mb
RSS double-counts pages shared between forked pool workers (torch, numpy);
PSS splits them fairly, so sum_pss_mb is the tree's real footprint.
The peak of each column is printed when the root exits.
"""
import argparse
import os
import sys
import time


def _children_map():
    kids = {}
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        try:
            with open('/proc/%s/stat' % name, 'r') as f:
                stat = f.read()
        except OSError:
            continue
        # comm (field 2) may contain spaces and parentheses
        rest = stat[stat.rfind(')') + 2:].split()
        try:
            ppid = int(rest[1])
        except (IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(name))
    return kids


def _tree(root):
    kids = _children_map()
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def _field_kb(path, field):
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith(field + ':'):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def sample(root):
    n = 0
    rss_sum = pss_sum = rss_max = 0
    for pid in _tree(root):
        rss = _field_kb('/proc/%d/status' % pid, 'VmRSS')
        if rss is None:
            continue
        n += 1
        rss_sum += rss
        rss_max = max(rss_max, rss)
        pss = _field_kb('/proc/%d/smaps_rollup' % pid, 'Pss')
        pss_sum += pss if pss is not None else rss
    return n, rss_sum / 1024.0, pss_sum / 1024.0, rss_max / 1024.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('root_pid', type=int)
    ap.add_argument('out')
    ap.add_argument('--interval', type=float, default=5.0)
    a = ap.parse_args(argv)
    t0 = time.time()
    peak = [0, 0.0, 0.0, 0.0]
    with open(a.out, 'w') as f:
        f.write('utc\telapsed_s\tn_procs\tsum_rss_mb\tsum_pss_mb\tmax_rss_mb\n')
        while os.path.exists('/proc/%d' % a.root_pid):
            n, rs, ps, rm = sample(a.root_pid)
            if n:
                for i, v in enumerate((n, rs, ps, rm)):
                    peak[i] = max(peak[i], v)
                f.write('%s\t%.0f\t%d\t%.0f\t%.0f\t%.0f\n'
                        % (time.strftime('%H:%M:%S', time.gmtime()),
                           time.time() - t0, n, rs, ps, rm))
                f.flush()
            time.sleep(a.interval)
    print('RSS-MONITOR root=%d duration_s=%.0f peak n_procs=%d sum_rss_mb=%.0f '
          'sum_pss_mb=%.0f max_single_rss_mb=%.0f'
          % (a.root_pid, time.time() - t0, peak[0], peak[1], peak[2], peak[3]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
