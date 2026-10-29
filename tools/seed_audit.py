#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed audit (WPDOC): every use of topology seeds in a window, across trees.

  nice -n 19 python3 tools/seed_audit.py --lo 200 --hi 219 \\
      --roots ~/MARL_run_* ~/MARL_code ~/MARL_code_remote ~/MARL_perf ~/MARL_watch \\
      --out output/seed_audit_200_219.json [--pt] [--git] [--git-log DUMP]

Read-only with respect to the audited content: the tool opens files and never
writes one, except its own JSON report (--out) and the log the caller
redirects.  Both MAY live inside a scanned root as long as their names are
planning-classified (anything containing `seed_audit`), which is how the
recorded run stores them (`~/MARL_run_9b/output/seed_audit_<lo>_<hi>.json`,
`~/MARL_run_9b/logs/seed_audit_<lo>_<hi>.log`): the report is written after the
walk, so it cannot influence its own run, and the next run buckets it as
planning.  A `--git-log` dump must NOT live inside a scanned root -- it carries
the non-planning diffs of ordinary source files, which the walk would read as
file content instead of as history -- so keep it in a directory excluded with
`--skip-dir` (the recorded run uses `~/MARL_run_9b/audit_history`, skipped).

The verdict has two halves and BOTH must hold for a CLEAN result (exit 0):

  * no hit    - no seed of [lo, hi] in result data, in a non-planning text
                line, in a file name or in scanned git history.  RESULT
                DATA IS ALL THREE CONTAINER SOURCES: pickles (step 4), JSON
                (step 3) AND torch checkpoints (step 5).  A checkpoint hit
                was recorded in the report but aggregated into no verdict
                field until 2026-09-23, so a seed of the window sitting in
                a checkpoint's metadata printed CLEAN and exited 0 while
                the same run's report carried pt[<file>]['hits'];
  * coverage  - every --roots entry existed, was walked AND held at least
                one file (an empty root is a failure, not a warning: a tree
                moved away or an unmounted path must not pass because the
                other roots still meet the aggregate floors), `errors` is
                empty, no pickle and no checkpoint failed to load, --pt was
                not aborted, every --min-* floor was met (aggregate,
                --min-per-root and the per-root --min-root NAME=N floors --
                and, when they come from --min-root-file, that file was
                read and parsed to at least one floor, so a floor-less run
                cannot pass by losing the file), every requested git scan
                produced output, and (unless --no-home-check) every
                directory matching $HOME/MARL* is among the roots --
                symbolic links included.
  The report also carries `synced_head`, the $TREE/.synced_head the run saw,
  because the live file is overwritten by the next sync and the record's
  "the history dump was captured at the HEAD the scanned tree was synced
  from" must stay checkable afterwards.

Exit status: 0 CLEAN, 3 TOUCHED (a hit), 4 INCOMPLETE (coverage refused).
An audit that scanned nothing can therefore never pass the freeze gate.

THE WINDOW IS A PARAMETER (--lo/--hi; the driver tools/run_seed_audit.sh
defaults to R2's block 200-219).  For the R3 confirmatory block the driver
adds two options, both off by default so that an R2 audit is what it was:
  --gate-result-data   exit 5 when a seed of the window is in RESULT DATA (a
                       pickle, a non-planning JSON file or a checkpoint), with
                       the files named under verdict.result_data_gate.  The R3
                       gate requires ZERO such hits; text, file-name and
                       history mentions keep exit 3 and are classified;
  --planning-marker S  a path substring classified as planning for this audit
                       (the R3 plan lives in confirmatory/chain_plan.py, its
                       tests and docs; at freeze 2 also the run's own S0
                       manifest).  Recorded in the report.  It can never
                       reclassify a pickle or a checkpoint.

What is scanned
  1. file names (every file under every root): integer tokens of the base
     name inside [lo, hi];
  2. text files (*.py *.sh *.log *.txt *.md *.yaml *.yml *.csv *.tsv *.sha256
     *.diff *.out *.nohup *.launcher *.ini *.cfg *.toml *.tex *.html *.js
     *.bak *.rst, and files without extension): on every line that mentions
     'seed', or carries a --seed-list /
     --seeds / --seed-range spec, or names a *_<a>_<b>.{pkl,json,log} file,
     the seed values named on that line (key=value, "seed N", CLI specs,
     a-b ranges and range(a, b) calls) are extracted and range-checked;
     every other standalone token in [lo, hi] on a 'seed' line is kept with
     its left context so that false positives can be classified.  A
     range(a, b) on a line that never says 'seed' is window-checked too,
     unless its loop variable names a tick, episode or graph position
     (`for tick in range(1, 361)`): such loops are listed in their own
     bucket and not counted;
  3. JSON files (parsed when <= --json-max bytes): every value under a key
     containing 'seed', and every dict keyed by integers (result exports are
     keyed scenario -> arm -> seed; manifests list topology_seed).  A file
     above --json-max is NOT line-scanned as text; it takes the regex
     fallback of this step instead, which reads the same two things -- a
     seed key's value whether it is a string, a number or a list literal
     (nested lists included), and every integer key.  A JSON file that
     FAILS TO PARSE is the one that is text-scanned, and is recorded as
     such.  The report names the path each file took per file ('scan' =
     'parsed' / 'regex-fallback' / 'text (parse failed)');
  4. pickles: loaded with a class-stubbing Unpickler (no sixg_sim import),
     deduplicated by sha256; the top-level 'seeds' entry, the provenance
     seeds and every int-keyed dict down to depth 6 are collected;
  5. --pt: torch checkpoints that are not per-agent adapted files
     (*_seed<N>.pt): values under keys containing 'seed' in their metadata;
  Symbolic links are never followed: os.walk does not descend into a
  symlinked directory and a symlinked file is not opened.  Every link met is
  counted and resolved, and one whose target is broken or lies outside every
  root is a named coverage warning, so content reachable only through a link
  cannot be missed silently.

  6. --git: `git log --all -p` of every root that has a .git entry (a
     directory or a worktree's .git file), run through the text patterns.  A
     git invocation that fails, or that produces no output, is a hard error
     and refuses the verdict -- an empty .git directory must not read as
     "nothing was ever committed".  --git-log FILE scans a captured
     `git log --all -p` dump instead, for a repository that lives on another
     host (the trackB worktree); the dump's sha256, byte count and commit
     count are recorded in the report.

Planning documents of this study (docs/RERUN_SPEC.md, the pre-registration,
docs/WP9_ANALYSIS.md, this audit's own outputs, ...) legitimately mention the
fresh range; hits in them are reported in a separate 'planning' bucket, never
as uses.  Directory names in SKIP_DIRS (plus --skip-dir) are not walked; the
unit tests' scratch directory 'tmp_tests' is one of them, because the
touched-tree test plants a seed of the window there on purpose.
"""
import argparse
import datetime
import gc
import glob
import hashlib
import io
import json
import os
import pickle
import re
import socket
import subprocess
import sys
import time

# 'tmp_tests' is the unit tests' scratch directory (D0.10, git-ignored): the
# audit's own touched-tree test writes a seed of the window there and removes
# it again, so a concurrent test run must not make a real tree look touched.
SKIP_DIRS = {'venv', '.venv', '.git', 'node_modules', '__pycache__',
             'site-packages', '.ipynb_checkpoints', 'tmp_tests'}
TEXT_EXTS = {'.py', '.sh', '.log', '.txt', '.md', '.yaml', '.yml', '.csv',
             '.tsv', '.sha256', '.diff', '.out', '.nohup', '.launcher',
             '.ini', '.cfg', '.toml', '.tex', '.html', '.js', '.bak', '.rst',
             '.gitignore', '.gitattributes', '.patch'}
BINARY_EXTS = {'.pt', '.pkl', '.png', '.gif', '.tar', '.docx', '.pptx',
               '.ppt', '.prof', '.pyc', '.so', '.zip', '.gz', '.npy', '.npz',
               '.pdf', '.jpg', '.jpeg', '.bin', '.pth', '.svg'}
PLANNING_MARKERS = ('docs/RERUN_SPEC.md', 'docs/PREREGISTRATION_R2',
                    'docs/PREREGISTRATION_R3',
                    'docs/COMPUTE_PLAN.md', 'docs/SEED_AUDIT_',
                    'docs/TRACKB_HANDOFF.md', 'docs/WP9_ANALYSIS.md',
                    'seed_audit', 'RERUN_SPEC')
PER_AGENT_PT = re.compile(r'_seed\d+\.pt$')

# ---- text patterns -------------------------------------------------------
_VAL = r'((?:[0-9][0-9,\s\-]*(?:and\s+)?)+)'
RE_KV = re.compile(r'(?i)\b[a-z_]*seeds?(?:_list|_range|_pool|_spec|list)?\b'
                   r'\s*(?:\([^)]*\))?["\']?\s*[=:]\s*["\']?\s*\[?\s*["\']?\s*' + _VAL)
RE_WORD = re.compile(r'(?i)\bseeds?\s+\(?' + _VAL)
RE_CLI = re.compile(r'--seed(?:-list|-range|s)?[=\s]+["\']?' + _VAL)
RE_RANGE_CALL = re.compile(r'range\(\s*(\d+)\s*,\s*(\d+)')
RE_FOR_RANGE = re.compile(r'\bfor\s+([A-Za-z_]\w*)\s+in\s+range\(')
# Loop variables that name a time step, an episode or a graph position.  A
# range() over one of them on a line that never says "seed" is a tick or
# episode loop: it is reported in its own bucket and not counted as a seed
# use.  Generic names (i, k, s, n, x, ...) stay counted.
BENIGN_LOOP_VARS = {'tick', 'ticks', 't', 'tk', 'step', 'steps', 'ep', 'epoch',
                    'epochs', 'episode', 'episodes', 'hop', 'hops', 'node',
                    'nodes', 'level', 'layer', 'it', 'iter', 'round', 'frame'}
RE_RANGE_NAME = re.compile(r'_(\d{2,4})_(\d{2,4})\.(?:pkl|json|log|txt)\b')
# The same, with the whole file name, so that a line which merely NAMES one of
# this study's planning outputs (`seed_audit_220_239.log` in the watchdog's
# status page, a command line in a plan) is classified as planning exactly as
# the file itself is -- otherwise the audit's own log name, quoted elsewhere,
# makes its own window look touched.
RE_RANGE_NAME_FULL = re.compile(
    r'([A-Za-z0-9][A-Za-z0-9_.+\-]*_(\d{2,4})_(\d{2,4})\.(?:pkl|json|log|txt))\b')
# `git log -p` file attribution: every hunk belongs to the file of the last
# `diff --git` header, so a diff of a planning document is classified as
# planning exactly as the file itself is.  A line before the first header
# (the commit message) belongs to no file and is never planning.
RE_DIFF_HDR = re.compile(r'^diff --git a/(.+?) b/(.+?)\s*$')
RE_COMMIT_HDR = re.compile(r'^commit [0-9a-f]{7,40}\b')
RE_NAME_SEED = re.compile(r'(?i)(?:seed|_s)[_\-]?(\d{1,7})(?=[_.\-\s"\'/)\]]|$)')
RE_TOKEN = re.compile(r'(?<![\d.])(\d+)(?![\d.])')
RE_DASH = re.compile(r'(\d+)\s*-\s*(\d+)')
RE_INT = re.compile(r'\d+')
# Large-JSON (above --json-max) regex path.  The pattern matches the KEY only;
# the value is then read by _json_value_at, which handles a string, a number
# and a list literal with nested lists.  The earlier pattern accepted only a
# bare integer or `[...]` up to the first `]`, so a string-valued seed key
# ("seed_range": "200-219") was not matched at all and a nested list lost
# everything after its first inner list -- silently, with the file not even
# listed as having seed content.  The fallback is now as strong as the parsed
# path for both shapes, and every file that took it is recorded ('scan').
RE_JSON_SEEDKEY = re.compile(r'"([^"]*[sS]eed[^"]*)"\s*:')
RE_JSON_INTKEY = re.compile(r'"(\d{1,7})"\s*:')
RE_JSON_NUMBER = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')

BENIGN_CONTEXTS = [
    ('sev=', re.compile(r'sev=$')),
    ('rescue=', re.compile(r'rescue=$')),
    ('Episode N', re.compile(r'(?i)episodes?\s*[=:#]?\s*$')),
    ('tick(s)', re.compile(r'(?i)ticks?\s*[=:(]?\s*$')),
    ('num_nodes', re.compile(r'num_nodes[^0-9]*$')),
    ('links=', re.compile(r'[a-z_]*links=$')),
    ('diff hunk', re.compile(r'@@\s*[-+]?\d*,?$|@@ [-+]$')),
    ('line number', re.compile(r'[:.]$')),
    ('range end', re.compile(r'range\(\s*\d+\s*,\s*$')),
]
OTHER_CAP = 3000          # unclassified tokens kept in the report
# Longest matched value string parsed for seed values.  The earlier limit of
# 160 characters silently dropped the tail of a long seed list, so a seed of
# the window named at the end of one was never extracted and the tree read
# CLEAN.  The limit is now far above any plausible seed list (a 20-seed list
# is ~100 characters, the 48-seed training spec ~300); reaching it is
# recorded as a hard error and refuses the verdict rather than truncating
# silently.
VALUE_CHARS = 20000


def _ints_from_value(val, cap=VALUE_CHARS):
    """Seed values named in a value string: singles and a-b ranges.
    Ranges are expanded only when 0 <= a <= b and b - a <= 5000.
    cap=None reads the whole string (the container paths, which record the
    full value elsewhere and must never drop its tail)."""
    out = set()
    if cap is not None:
        val = val[:cap]
    for a, b in RE_DASH.findall(val):
        a, b = int(a), int(b)
        if a <= b and b - a <= 5000:
            out.update(range(a, b + 1))
    stripped = RE_DASH.sub(' ', val)
    for n in RE_INT.findall(stripped):
        out.add(int(n))
    return out


def _fmt_ranges(seeds):
    seeds = sorted(set(int(s) for s in seeds))
    if not seeds:
        return '(none)'
    parts, start, prev = [], seeds[0], seeds[0]
    for s in seeds[1:]:
        if s == prev + 1:
            prev = s
            continue
        parts.append('%d-%d' % (start, prev) if prev > start else str(start))
        start = prev = s
    parts.append('%d-%d' % (start, prev) if prev > start else str(start))
    return ', '.join(parts)


def _intlike(k):
    if isinstance(k, bool):
        return False
    if isinstance(k, int):
        return True
    return isinstance(k, str) and k.isdigit()


def _is_planning(rel):
    rel = rel.replace('\\', '/')
    return any(m in rel for m in PLANNING_MARKERS)


# ---- class-stubbing unpickler ------------------------------------------
class _Stub(object):
    """Stands in for any class the audit host cannot (and need not) import."""

    def __init__(self, *a, **k):
        self._args = a

    def __setstate__(self, state):
        self.__dict__['_state'] = state

    def __call__(self, *a, **k):
        return _Stub(*a)


SAFE_MODULES = ('builtins', 'collections', 'numpy', 'copyreg', '_codecs',
                'datetime', 'decimal', 'fractions', 'statistics', 'torch',
                'pathlib', 'enum')


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] in SAFE_MODULES:
            try:
                return super(StubUnpickler, self).find_class(module, name)
            except Exception:
                return _Stub
        return _Stub

    def persistent_load(self, pid):
        return None


def load_pickle_stub(path):
    with open(path, 'rb') as f:
        return StubUnpickler(f).load()


# ---- generic container walk (JSON, pickles, checkpoints) ------------------
def new_walk_out():
    """The accumulator of a container walk.

    'intkeyed' and 'seedkeys' are the REPORT's summaries and are capped (400
    integers, 120 characters per value, WALK_REPORT_CAP entries) so that a
    report stays readable.  'ints' is the untruncated set of every integer
    the walk found, and it -- never the summaries -- is what the window test
    reads.  'truncated' is non-empty only if the walk stopped early; the
    caller turns every message in it into an `errors` entry, which refuses
    coverage, so a walk that did not finish can never read as CLEAN."""
    return {'intkeyed': [], 'seedkeys': [], 'ints': set(), 'truncated': []}


# The hit path has NO depth and NO breadth limit (see walk_seeds).  These two
# bound only what a single file may cost: WALK_REPORT_CAP entries in the
# report's own lists, and WALK_NODE_LIMIT distinct containers visited --
# reaching the latter is recorded and REFUSES COVERAGE, it is never silent.
WALK_REPORT_CAP = 5000
WALK_NODE_LIMIT = 4000000


def walk_seeds(obj, path, out):
    """Collect (path, keys) of int-keyed dicts and values under 'seed' keys.

    NO CAP ON THE HIT PATH.  This walk used to return at depth > 6 and to
    stop after the 1000th container element of a list, and a seed key below
    either cap was dropped with no `errors` entry, no `long_values` entry,
    no coverage warning and no coverage failure -- the file was not even
    listed as carrying seed content.  A 1440-tick harness run makes a list
    of more than 1000 per-tick dicts an ordinary shape, and the large-JSON
    regex fallback (which has neither limit) was therefore the STRONGER of
    the two paths: the parsed path could miss what the fallback found.

    The walk is iterative, so no nesting depth can exhaust the interpreter
    stack, and it remembers the containers it has entered by id(), so a
    cyclic or shared structure is walked once and cannot loop.  Every
    container reached is held alive by `obj` for the duration of the call,
    so an id cannot be reused underneath the set.  Children are pushed in
    reverse so that the traversal order, and hence the report's lists, are
    the depth-first pre-order the recursive version produced.

    Only WALK_NODE_LIMIT remains, as a cost bound, and reaching it appends a
    message to out['truncated'] which the caller records as an error."""
    stack = [(obj, path)]
    seen = set()
    nodes = 0
    while stack:
        cur, p = stack.pop()
        if isinstance(cur, (dict, list, tuple, _Stub)):
            if id(cur) in seen:
                continue                    # cycle, or a shared substructure
            seen.add(id(cur))
        else:
            continue                        # a scalar reached as a container
        nodes += 1
        if nodes > WALK_NODE_LIMIT:
            out['truncated'].append(
                'container walk stopped at %s after %d containers '
                '(WALK_NODE_LIMIT); seed values may be missing'
                % (p or '/', WALK_NODE_LIMIT))
            return
        if isinstance(cur, _Stub):
            st = cur.__dict__.get('_state')
            if isinstance(st, (dict, list, tuple, _Stub)):
                stack.append((st, p + '<stub>'))
            continue
        if isinstance(cur, dict):
            keys = list(cur.keys())
            if keys and all(_intlike(k) for k in keys):
                ks = sorted(int(k) for k in keys)
                if len(out['intkeyed']) < WALK_REPORT_CAP:
                    out['intkeyed'].append({'path': p, 'n': len(ks), 'keys': ks[:400]})
                out['ints'].update(ks)      # every key, not the first 400
            children = []
            for k, v in cur.items():
                ks = str(k)
                if 'seed' in ks.lower():
                    if len(out['seedkeys']) < WALK_REPORT_CAP:
                        out['seedkeys'].append({'path': p + '/' + ks, 'value': _summ(v)})
                    out['ints'].update(_value_ints(v))
                if isinstance(v, (dict, list, tuple, _Stub)):
                    children.append((v, p + '/' + ks))
            stack.extend(reversed(children))
        else:                               # list or tuple: every element
            children = []
            for i, v in enumerate(cur):
                if isinstance(v, (dict, list, tuple, _Stub)):
                    children.append((v, p + '[%d]' % i))
            stack.extend(reversed(children))


def _summ(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return [v]
    if isinstance(v, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        return sorted(set(v))[:400]
    if isinstance(v, dict):
        ks = [int(k) for k in v if _intlike(k)]
        return sorted(ks)[:400] if ks else str(list(v)[:10])[:120]
    return str(v)[:120]


def _value_ints(v, depth=0):
    """Every integer a value under a 'seed' key names -- UNTRUNCATED.

    _summ() is the report's copy of the same value and is deliberately short
    (400 integers, 120 characters).  Reading the hit test off that copy is how
    a seed named past character 120 of a string value -- `"seed_spec": "...,
    205 confirmatory"` -- was dropped with no error, no long_values entry and
    no coverage warning, so the tree read CLEAN.  Nothing here truncates: a
    string is read whole with _ints_from_value, which is the same extractor
    the text path uses, so `"seed_range": "190-230"` names 190..230 exactly
    as that spec does on a script line (the previous integer-only reading
    named 190 and 230 and would have missed the window entirely).

    Recursion follows the shape of _summ: a list is read element by element
    (nested lists included), a dict contributes its integer-like keys (its
    values are reached by walk_seeds itself, which registers every int-keyed
    dict and every nested 'seed' key on its own)."""
    if v is None or isinstance(v, bool):
        return set()
    if isinstance(v, int):
        return {v}
    if isinstance(v, float):
        # a float is not a seed, but a seed written as 205.0 is.  Reading the
        # DIGITS of a float is what the old path did (str(v)[:120] through a
        # plain \d+), so every standard deviation under a key such as
        # `sd_across_seeds` contributed integers that were never seeds:
        # 6.064136589897359 named 6, and merged_results.json alone put 43 such
        # fragments into the "seeds ever used" inventory, where the record
        # then explained them as flow and node identifiers.
        return {int(v)} if v.is_integer() and abs(v) < 1e15 else set()
    if isinstance(v, (list, tuple)) and depth < 6:
        out, ints_only = set(), True
        for x in v:
            if isinstance(x, int) and not isinstance(x, bool):
                out.add(x)
            else:
                ints_only = False
        if ints_only:
            return out
        out = set()
        for x in v:
            out |= _value_ints(x, depth + 1)
        return out
    if isinstance(v, dict) and depth < 6:
        ks = set(int(k) for k in v if _intlike(k))
        return ks if ks else _ints_from_value(str(list(v)), cap=None)
    s = v if isinstance(v, str) else str(v)
    return _ints_from_value(s, cap=None)


def _json_value_at(txt, i):
    """The JSON value at or after txt[i], as source text, for the large-file
    fallback path: a string's contents, a number, or a list literal with its
    nested lists.  An object is skipped -- its own integer keys and 'seed'
    keys are matched by the same two patterns over the whole text.  None means
    there was no value to read (a truncated file), which the caller records."""
    n = len(txt)
    while i < n and txt[i] in ' \t\r\n':
        i += 1
    if i >= n:
        return None
    c = txt[i]
    if c == '"':
        j = i + 1
        while j < n:
            if txt[j] == '\\':
                j += 2
                continue
            if txt[j] == '"':
                return txt[i + 1:j]
            j += 1
        return None
    if c == '[':
        depth, j, instr = 0, i, False
        while j < n:
            ch = txt[j]
            if instr:
                if ch == '\\':
                    j += 2
                    continue
                if ch == '"':
                    instr = False
            elif ch == '"':
                instr = True
            elif ch in '[{':
                depth += 1
            elif ch in ']}':
                depth -= 1
                if depth == 0:
                    return txt[i:j + 1]
            j += 1
        return None
    if c == '{':
        return ''                      # object: its keys are matched separately
    m = RE_JSON_NUMBER.match(txt, i)
    return m.group(0) if m else ''     # true/false/null: no integer


def _intkeyed_inventory(out, cap=40):
    """'path: keys' lines, one per int-keyed dict path and per 'seed' key, so
    that a report explains where every listed integer comes from."""
    seen, lines = {}, []
    for e in out['intkeyed']:
        seen.setdefault(e['path'] or '/', set()).update(e['keys'])
    for e in out['seedkeys']:
        v = e['value']
        if isinstance(v, list):
            seen.setdefault(e['path'], set()).update(v)
    for path in sorted(seen):
        lines.append('%s: %s' % (path, _fmt_ranges(seen[path])))
        if len(lines) >= cap:
            lines.append('... (%d paths)' % len(seen))
            break
    return lines


def _walk_truncation_errors(out, rel):
    """Every reason a container walk did not finish, as `errors` lines.

    A walk that stopped early has not cleared the file, so the run must not
    read as complete: `errors` being non-empty is already a coverage failure
    (exit 4).  Silence is the one option this tool's premise rules out."""
    return ['%s: %s' % (rel, m) for m in out.get('truncated') or []]


def _ints_of_walk(out):
    """The integers of a walk, for the window test.

    This used to re-read the REPORT's summaries: the first 400 keys of an
    int-keyed dict, and -- for a string value -- the first 50 integers of a
    copy already cut to 120 characters.  Three silent truncations in the one
    place that decides whether a tree is touched.  walk_seeds now accumulates
    every integer in out['ints'] as it goes."""
    return set(out['ints'])


# ---- the audit ------------------------------------------------------------
class Audit(object):
    def __init__(self, lo, hi, roots, json_max=64 * 1024 * 1024, do_pt=False,
                 do_git=False, verbose=False, false_positives=(), skip_dirs=(),
                 git_logs=(), min_counts=None, home_check=True, argv=None,
                 home_glob='~/MARL*', git_no_repo=(), min_per_root=0,
                 min_root=None, min_root_file=None, tree=None,
                 planning_markers=(), gate_result_data=False):
        self.lo, self.hi = int(lo), int(hi)
        # --planning-marker: path substrings classified as PLANNING for THIS
        # audit, in addition to PLANNING_MARKERS (the R3 block's plan lives in
        # other files than the R2 block's did).  Each one is an explicit,
        # recorded classification -- it is in the command and in the report --
        # and it can only move text lines, JSON files and file names: a hit
        # in a pickle or in a checkpoint is result data whatever its path.
        self.planning_markers = tuple(m.replace('\\', '/') for m in planning_markers if m)
        # --gate-result-data (the R3 freeze gate): a window seed in RESULT
        # DATA is a hard stop of its own (exit 5), distinct from text, name
        # or history mentions that remain to be classified (exit 3).
        self.gate_result_data = bool(gate_result_data)
        self.roots = [os.path.abspath(os.path.expanduser(r)) for r in roots]
        self.json_max = json_max
        self.skip_dirs = set(SKIP_DIRS) | set(d for d in skip_dirs if d)
        self.do_pt, self.do_git, self.verbose = do_pt, do_git, verbose
        self.git_logs = [os.path.abspath(os.path.expanduser(p)) for p in git_logs if p]
        # roots whose .git entry is known NOT to be a repository (~/MARL_code
        # holds an empty .git directory).  Declaring one turns the hard error
        # into a recorded warning, so the exception is named in the command
        # and in the report instead of passing silently.
        self.git_no_repo = set(os.path.abspath(os.path.expanduser(p))
                               for p in (git_no_repo or ()) if p)
        # minimum coverage the run must reach for a CLEAN verdict; the freeze
        # command passes the counts of the recorded audit so that a run which
        # scanned less than the record cannot pass the gate
        self.min_counts = dict(min_counts or {})
        # per-root floors.  The aggregate --min-* floors cannot see a single
        # root that lost its content: the recorded run has ~966 files of slack
        # over --min-files, more than the whole of ~/MARL_run_9c (239) or
        # ~/MARL_watch (12).  --min-per-root N is a floor for every root;
        # --min-root NAME=N pins one root by basename.
        self.min_per_root = int(min_per_root or 0)
        self.min_root = {k: int(v) for k, v in (min_root or {}).items()}
        # --min-root-file: the floors are read BY THE TOOL, so the report
        # proves they were applied.  The driver used to expand the file into
        # --min-root arguments in a `while read` loop under `set -u` without
        # `set -e`: a missing, renamed or unreadable file left the array
        # empty, the loop's failure was not checked, the audit ran with no
        # per-root floor at all and could still exit 0.  The only trace was
        # the absence of --min-root entries in the report's argv, which
        # nothing checked.  Here an unreadable or empty file is a COVERAGE
        # FAILURE, so the freeze gate reads it off the report.
        self.min_root_file = min_root_file
        # the tree this audit is run from; its .synced_head is recorded with
        # the verdict because the live file is overwritten by the next sync
        # and the record's "the dump was captured at the HEAD the scanned
        # tree was synced from" then stops being reconstructable
        self.tree = os.path.abspath(os.path.expanduser(tree or os.getcwd()))
        self.home_check = bool(home_check)
        self.home_glob = home_glob
        self.pt_aborted = False
        # substrings of line snippets that a human has classified as not a
        # seed use (e.g. "seed 42, 200 island ticks"); such hits are kept in
        # their own bucket and do not count against the verdict
        self.false_positives = tuple(s for s in false_positives if s)
        self.rep = {
            'window': [self.lo, self.hi], 'roots': self.roots,
            'host': socket.gethostname(),
            'argv': list(argv) if argv is not None else None,
            'started': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'false_positive_patterns': list(self.false_positives),
            'skip_dirs': sorted(self.skip_dirs),
            'requested': {'pt': bool(do_pt), 'git': bool(do_git),
                          'git_logs': list(self.git_logs), 'home_check': self.home_check,
                          'git_no_repo': sorted(self.git_no_repo),
                          'min_counts': dict(self.min_counts),
                          'min_per_root': self.min_per_root,
                          'min_root': dict(self.min_root),
                          'min_root_file': self.min_root_file,
                          'tree': self.tree},
            'min_root_file': None,          # filled by load_min_root_file()
            'synced_head': None,            # filled by record_synced_head()
            'counts': {'files': 0, 'text': 0, 'json_parsed': 0, 'json_scanned': 0,
                       'pickles': 0, 'pickles_loaded': 0, 'pickles_duplicate': 0,
                       'pickles_failed': 0, 'pt': 0, 'pt_failed': 0, 'pt_skipped': 0,
                       'skipped_binary': 0, 'git_roots_scanned': 0, 'git_log_lines': 0,
                       'symlinks': 0, 'symlink_dirs': 0},
            'per_root': {},
            'home_marl_dirs': [], 'home_marl_unscanned': [], 'home_marl_symlinks': [],
            'symlinks': [], 'symlinks_outside_roots': [],
            'filename_hits': [], 'text_hits': [], 'planning_hits': [],
            'false_positive_hits': [],
            'range_nonseed_benign': [],     # tick/episode loops, never counted
            'broad_tokens': {'histogram': {}, 'other': []},
            'json': {}, 'pickles': {}, 'pt': {}, 'git_hits': [],
            'git_roots': [], 'git_logs': [],
            'long_values': [],              # matched value strings at VALUE_CHARS
            'text_value_sources': {},
            'errors': [], 'coverage_warnings': [],
        }
        self.seeds_data = set()      # from pickles / JSON int keys / seed keys
        self.seeds_text = set()      # named on text lines (kv, word, cli), planning excluded
        self.seeds_names = set()     # from file names
        self._digests = {}
        # recorded only when used, so a default (R2) report is what it was
        if self.planning_markers:
            self.rep['requested']['planning_markers'] = list(self.planning_markers)
        if self.gate_result_data:
            self.rep['requested']['gate_result_data'] = True

    # -- helpers
    def is_planning(self, rel):
        """A planning path: the module's PLANNING_MARKERS, or one of this
        audit's own --planning-marker substrings."""
        if _is_planning(rel):
            return True
        rel = rel.replace('\\', '/')
        return any(m in rel for m in self.planning_markers)

    def load_min_root_file(self):
        """Read the per-root floors from a file, and record what was read.

        docs/SEED_AUDIT_200_219.md 5 item 3 requires "every per-root floor
        of tools/seed_audit_floors.txt"; until 2026-09-23 the only thing that
        carried that requirement was a shell loop in the driver whose failure
        was invisible, and item 8 accepted a run purely because it came from
        that driver.  With the floors read here, the report names the file,
        its sha256 and every floor parsed from it, and a file that is missing,
        unreadable, unparsable or empty refuses the verdict (exit 4)."""
        p = self.min_root_file
        if not p:
            return
        p = os.path.abspath(os.path.expanduser(p))
        ent = {'path': p, 'sha256': None, 'floors': {}, 'ok': False}
        self.rep['min_root_file'] = ent
        try:
            with open(p, 'rb') as f:
                raw = f.read()
        except OSError as e:
            self.rep['errors'].append('--min-root-file %s: %s' % (p, e))
            return
        ent['sha256'] = hashlib.sha256(raw).hexdigest()
        bad = []
        for ln, line in enumerate(raw.decode('utf-8', 'replace').splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, sep, n = line.partition('=')
            name, n = name.strip(), n.strip()
            if not sep or not name or not n.isdigit():
                bad.append('%d: %r' % (ln, line[:60]))
                continue
            ent['floors'][name] = int(n)
            self.min_root[name] = int(n)
        if bad:
            self.rep['errors'].append(
                '--min-root-file %s: %d unparsable line(s) (expected NAME=N): %s'
                % (p, len(bad), '; '.join(bad[:5])))
            return
        if not ent['floors']:
            self.rep['errors'].append(
                '--min-root-file %s: no per-root floor parsed; the audit would '
                'have run with no --min-root at all' % p)
            return
        ent['ok'] = True
        self.rep['requested']['min_root'] = dict(self.min_root)

    def record_synced_head(self):
        """Copy $TREE/.synced_head into the report.

        docs/SEED_AUDIT_200_219.md 2.2 pointed a reviewer at the LIVE file to
        confirm that the history dump and the scanned tree were at the same
        HEAD.  The next sync overwrites it, and the equality -- the central
        evidence of that section -- then cannot be reconstructed from the
        server at all.  Recorded here, it is frozen with the verdict."""
        p = os.path.join(self.tree, '.synced_head')
        ent = {'path': p, 'value': None, 'mtime_utc': None, 'error': None}
        self.rep['synced_head'] = ent
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                ent['value'] = f.read().strip()
            ent['mtime_utc'] = datetime.datetime.fromtimestamp(
                os.path.getmtime(p), datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        except OSError as e:
            # not every tree has one (the unit tests' scratch trees do not);
            # this is recorded, not a coverage failure
            ent['error'] = str(e)

    def in_window(self, values):
        return sorted(v for v in values if self.lo <= v <= self.hi)

    def _rel(self, path):
        for r in self.roots:
            if path.startswith(r):
                return os.path.basename(r) + path[len(r):]
        return path

    def log(self, *a):
        if self.verbose:
            print('[audit]', *a, file=sys.stderr, flush=True)

    # -- walk
    def record_symlink(self, path, is_dir=False):
        """A symbolic link is never followed.  It is counted and resolved.

        The completeness claim used to read "directory names <SKIP_DIRS> are
        not walked; nothing else is", while every link was dropped by a bare
        `continue`, with no count and no record.  A link whose target is
        broken, or lies outside every scanned root, is content the audit did
        not read, so it becomes a named coverage warning; a link into a
        scanned root costs nothing, because the target is walked on its own."""
        rel = self._rel(path)
        try:
            target = os.path.realpath(path)
        except OSError:
            target = None
        self.rep['counts']['symlinks'] += 1
        if is_dir:
            self.rep['counts']['symlink_dirs'] += 1
        inside = bool(target) and any(
            target == r or target.startswith(r + os.sep) for r in self.roots)
        entry = {'path': rel, 'target': target, 'dir': bool(is_dir),
                 'inside_roots': inside,
                 'broken': not (target and os.path.exists(target))}
        if len(self.rep['symlinks']) < 4000:
            self.rep['symlinks'].append(entry)
        if entry['broken'] or not inside:
            self.rep['symlinks_outside_roots'].append(entry)

    def check_home_roots(self):
        """Every directory matching $HOME/MARL* must be among --roots.

        The roots of the freeze command used to be a frozen literal list; a
        tree created after it was written (~/MARL_run_9c, 2026-09-23) was then
        silently left unscanned while the run still exited 0.  The check is
        active whenever at least one root IS one of those directories (so the
        unit tests, which scan temporary trees, are unaffected) and can be
        turned off with --no-home-check for an ad-hoc single-tree audit."""
        pattern = os.path.abspath(os.path.expanduser(self.home_glob))
        # os.path.isdir follows the link, so a SYMLINKED tree
        # (~/MARL_run_10 -> /data/...) is a directory that must be a root like
        # any other.  It used to be excluded here and, being a symlink, was
        # not walked either: neither required nor scanned, and coverage ok.
        dirs = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
        self.rep['home_marl_dirs'] = dirs
        self.rep['home_marl_symlinks'] = [d for d in dirs if os.path.islink(d)]
        if not self.home_check:
            return
        roots = set(self.roots) | set(os.path.realpath(r) for r in self.roots)
        if not (roots & (set(dirs) | set(os.path.realpath(d) for d in dirs))):
            return                      # not a home audit (unit tests, ad-hoc)
        self.rep['home_marl_unscanned'] = [
            d for d in dirs if d not in roots and os.path.realpath(d) not in roots]

    def run(self):
        t0 = time.time()
        self.load_min_root_file()
        self.record_synced_head()
        self.check_home_roots()
        pickles, pts = [], []
        for root in self.roots:
            n_root = 0
            if not os.path.isdir(root):
                self.rep['errors'].append('missing root %s' % root)
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in self.skip_dirs]
                for d in dirnames:
                    dp = os.path.join(dirpath, d)
                    if os.path.islink(dp):
                        # os.walk does not descend into a symlinked directory
                        self.record_symlink(dp, is_dir=True)
                for fn in filenames:
                    path = os.path.join(dirpath, fn)
                    if os.path.islink(path):
                        self.record_symlink(path)
                        continue
                    if not os.path.isfile(path):
                        continue
                    n_root += 1
                    self.rep['counts']['files'] += 1
                    self.scan_name(path)
                    ext = os.path.splitext(fn)[1].lower()
                    try:
                        if ext == '.pkl':
                            pickles.append(path)
                        elif ext == '.pt':
                            pts.append(path)
                        elif ext == '.json':
                            self.scan_json(path)
                        elif ext in BINARY_EXTS:
                            self.rep['counts']['skipped_binary'] += 1
                        elif ext in TEXT_EXTS or (ext == '' and self._looks_text(path)):
                            self.scan_text(path)
                        elif self._looks_text(path):
                            self.scan_text(path)
                        else:
                            self.rep['counts']['skipped_binary'] += 1
                    except OSError as e:
                        # a file that vanished or became unreadable while the
                        # tree is in use (another job's temporary output)
                        self.rep['errors'].append('%s: %s' % (self._rel(path), e))
            self.rep['per_root'][root] = n_root
            self.log('walked', root, n_root, 'files', '%.0fs' % (time.time() - t0))

        def _size(p):
            try:
                return os.path.getsize(p)
            except OSError:
                return -1
        for p in sorted(pickles, key=_size):
            self.scan_pickle(p)
        self.log('pickles done', '%.0fs' % (time.time() - t0))
        if self.do_pt:
            for i, p in enumerate(pts):
                if not self.do_pt:          # aborted (torch unavailable)
                    self.rep['counts']['pt_skipped'] = len(pts) - i
                    break
                try:
                    self.scan_pt(p)
                except OSError as e:
                    self.rep['errors'].append('%s: %s' % (self._rel(p), e))
            self.log('checkpoints done', '%.0fs' % (time.time() - t0))
        if self.do_git:
            for root in self.roots:
                self.scan_git(root)
        for p in self.git_logs:
            self.scan_git_log(p)
        self.finish(t0)
        return self.rep

    @staticmethod
    def _looks_text(path):
        try:
            with open(path, 'rb') as f:
                head = f.read(8192)
        except OSError:
            return False
        return b'\0' not in head

    # -- 1. names
    def scan_name(self, path):
        base = os.path.basename(path)
        toks = [int(t) for t in RE_INT.findall(base)]
        named = set()
        for m in RE_NAME_SEED.finditer(base):
            named.add(int(m.group(1)))
        for a, b in RE_RANGE_NAME.findall(base):
            a, b = int(a), int(b)
            if a <= b and b - a <= 5000:
                named.update(range(a, b + 1))
        rel = self._rel(path)
        planning = self.is_planning(rel)
        # the planning exclusion of seeds_data/seeds_text applies here too:
        # the audit's own report and log names (seed_audit_200_219.json)
        # must not enter the "seeds ever used" inventory as file-name seeds
        if not planning:
            self.seeds_names.update(named)
        hits = self.in_window(set(toks) | named)
        if hits:
            entry = {'path': rel, 'tokens': hits, 'seedlike': bool(self.in_window(named)),
                     'planning': planning}
            self.rep['filename_hits'].append(entry)

    # -- 2. text
    def scan_text(self, path, source=None, lines=None):
        rel = source or self._rel(path)
        planning = self.is_planning(rel)
        self.rep['counts']['text'] += 1
        try:
            if lines is None:
                fh = open(path, 'r', encoding='utf-8', errors='replace')
            else:
                fh = lines
            for ln, line in enumerate(fh, 1):
                self.scan_line(rel, ln, line.rstrip('\n'), planning)
            if lines is None:
                fh.close()
        except OSError as e:
            self.rep['errors'].append('%s: %s' % (rel, e))

    def scan_line(self, rel, ln, line, planning):
        low = line.lower()
        seedy = 'seed' in low
        named_range = ('.pkl' in low or '.json' in low or '.log' in low or '.txt' in low)
        rangey = 'range(' in line
        if not (seedy or named_range or rangey):
            return
        values, kinds = set(), []
        window_only = set()      # checked against the window, not "seeds referenced"
        if seedy:
            for kind, rx in (('kv', RE_KV), ('word', RE_WORD), ('cli', RE_CLI)):
                for m in rx.finditer(line):
                    raw = m.group(1)
                    if len(raw) >= VALUE_CHARS:
                        # the parse limit was reached: the tail of this value
                        # is NOT extracted, so the line cannot be cleared
                        self.rep['long_values'].append(
                            {'path': rel, 'line': ln, 'kind': kind, 'chars': len(raw)})
                        self.rep['errors'].append(
                            '%s:%d: %s value of %d characters exceeds the %d-character '
                            'parse limit; seed values may be missing'
                            % (rel, ln, kind, len(raw), VALUE_CHARS))
                    v = _ints_from_value(raw)
                    if v:
                        values |= v
                        kinds.append(kind)
            for m in RE_NAME_SEED.finditer(line):
                values.add(int(m.group(1)))
                kinds.append('name')
        if rangey:
            # range(a, b) on ANY line (a loop over seeds need not say "seed");
            # on a non-seed line the values only count as window hits, unless
            # the loop variable names a tick/episode/position (BENIGN_LOOP_VARS):
            # such a loop is listed in its own bucket and never counted
            fm = RE_FOR_RANGE.search(line)
            loop_var = fm.group(1) if fm else None
            benign_loop = (not seedy and loop_var is not None
                           and loop_var.lower() in BENIGN_LOOP_VARS)
            for a, b in RE_RANGE_CALL.findall(line):
                a, b = int(a), int(b)
                if not (a < b and b - a <= 5000):
                    continue
                if seedy:
                    values.update(range(a, b))
                    kinds.append('range()')
                elif benign_loop:
                    w = self.in_window(range(a, b))
                    if w:
                        self.rep['range_nonseed_benign'].append(
                            {'path': rel, 'line': ln, 'var': loop_var,
                             'values': [w[0], w[-1]], 'snippet': line.strip()[:200],
                             'planning': planning})
                else:
                    window_only.update(range(a, b))
                    kinds.append('range()-nonseed')
        plan_vals = set()        # from the name of a planning output only
        if named_range:
            for m in RE_RANGE_NAME_FULL.finditer(line):
                full, a, b = m.group(1), int(m.group(2)), int(m.group(3))
                if not (a <= b and b - a <= 5000):
                    continue
                if self.is_planning(full):
                    plan_vals.update(range(a, b + 1))
                else:
                    values.update(range(a, b + 1))
                    kinds.append('range-name')
        snippet = line.strip()[:200]
        false_pos = bool(self.false_positives) and any(s in snippet for s in self.false_positives)
        if values and not planning and not false_pos:
            self.seeds_text.update(values)
            src = self.rep['text_value_sources']
            for v in values:
                if v <= 1000 and v not in src:
                    src[v] = '%s:%d %s' % (rel, ln, snippet[:120])
        hits = self.in_window(values | window_only)
        if hits:
            entry = {'path': rel, 'line': ln, 'kinds': sorted(set(kinds)),
                     'values': hits, 'snippet': snippet}
            if planning:
                self.rep['planning_hits'].append(entry)
            elif false_pos:
                self.rep['false_positive_hits'].append(entry)
            else:
                self.rep['text_hits'].append(entry)
        plan_hits = self.in_window(plan_vals - values)
        if plan_hits and not planning:
            # a line outside the planning documents that names one of their
            # outputs (the watchdog's status page names the newest audit log)
            self.rep['planning_hits'].append(
                {'path': rel, 'line': ln, 'kinds': ['range-name (planning output)'],
                 'values': plan_hits, 'snippet': snippet})
        if seedy:
            for m in RE_TOKEN.finditer(line):
                n = int(m.group(1))
                if n in hits or not (self.lo <= n <= self.hi):
                    continue
                ctx = line[max(0, m.start() - 18):m.start()]
                label = 'other'
                for name, rx in BENIGN_CONTEXTS:
                    if rx.search(ctx):
                        label = name
                        break
                h = self.rep['broad_tokens']['histogram']
                h[label] = h.get(label, 0) + 1
                if label == 'other' and len(self.rep['broad_tokens']['other']) < OTHER_CAP:
                    self.rep['broad_tokens']['other'].append(
                        {'path': rel, 'line': ln, 'token': n, 'ctx': ctx.strip()[-18:],
                         'snippet': snippet, 'planning': planning})

    # -- 3. JSON
    def scan_json(self, path):
        rel = self._rel(path)
        planning = self.is_planning(rel)
        size = os.path.getsize(path)          # OSError handled by the caller
        out = new_walk_out()
        scan = 'parsed'
        if size <= self.json_max:
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    obj = json.load(f)
                self.rep['counts']['json_parsed'] += 1
                walk_seeds(obj, '', out)
                del obj
            except Exception as e:
                self.rep['errors'].append('%s: json parse failed (%s); scanned as text' % (rel, e))
                self.rep['json'][rel] = {
                    'size': size, 'seed_values': '(scanned as text)', 'n_intkeyed': 0,
                    'n_seedkeys': 0, 'intkeyed_paths': [], 'hits': [],
                    'planning': planning, 'detail': None, 'scan': 'text (parse failed)'}
                self.scan_text(path)
                return
        else:
            # Fallback for a file too large to parse.  It must read the same
            # two things as the parsed path, or the report shows a scan that
            # was quietly weaker: a seed key's value (string, number or list
            # literal, nested lists included) and every integer key.
            scan = 'regex-fallback'
            self.rep['counts']['json_scanned'] += 1
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    txt = f.read()
            except OSError as e:
                self.rep['errors'].append('%s: %s' % (rel, e))
                return
            seedvals, intkeys, unread = {}, set(), []
            for m in RE_JSON_SEEDKEY.finditer(txt):
                raw = _json_value_at(txt, m.end())
                if raw is None:
                    unread.append(m.group(1))
                    continue
                seedvals.setdefault(m.group(1), set()).update(
                    _ints_from_value(raw, cap=None))
            for k in RE_JSON_INTKEY.findall(txt):
                intkeys.add(int(k))
            del txt
            if unread:
                self.rep['errors'].append(
                    '%s: the regex fallback could not read the value of %d seed key(s) '
                    '(unterminated string or list): %s'
                    % (rel, len(unread), ', '.join(sorted(set(unread))[:5])))
            for k, vs in sorted(seedvals.items()):
                out['seedkeys'].append({'path': '/regex/' + k, 'value': sorted(vs)[:400]})
                out['ints'].update(vs)
            if intkeys:
                out['intkeyed'].append({'path': '/regex/int-keys', 'n': len(intkeys),
                                        'keys': sorted(intkeys)[:400]})
                out['ints'].update(intkeys)
        self.rep['errors'].extend(_walk_truncation_errors(out, rel))
        ints = _ints_of_walk(out)
        if not planning:      # the audit's own reports name the window
            self.seeds_data.update(i for i in ints if i <= 100000)
        hits = self.in_window(ints)
        detail = None
        if hits:
            detail = {'intkeyed': out['intkeyed'], 'seedkeys': out['seedkeys'],
                      'ints_in_window': hits}
        self.rep['json'][rel] = {
            'size': size, 'seed_values': _fmt_ranges(i for i in ints if i <= 100000),
            'n_intkeyed': len(out['intkeyed']), 'n_seedkeys': len(out['seedkeys']),
            'intkeyed_paths': _intkeyed_inventory(out),
            'hits': hits, 'planning': planning, 'scan': scan,
            'detail': detail,
        }
        gc.collect()

    # -- 4. pickles
    def scan_pickle(self, path):
        rel = self._rel(path)
        self.rep['counts']['pickles'] += 1
        h = hashlib.sha256()
        try:
            size = os.path.getsize(path)
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
        except OSError as e:
            self.rep['errors'].append('%s: %s' % (rel, e))
            return
        digest = h.hexdigest()
        entry = {'size': size, 'sha256': digest[:16], 'duplicate_of': None,
                 'seeds_key': None, 'seed_values': None, 'hits': [], 'error': None}
        if digest in self._digests:
            first = self._digests[digest]
            entry['duplicate_of'] = first
            src = self.rep['pickles'][first]
            entry.update(seeds_key=src['seeds_key'], seed_values=src['seed_values'],
                         hits=src['hits'], error=src['error'])
            self.rep['counts']['pickles_duplicate'] += 1
            self.rep['pickles'][rel] = entry
            return
        self._digests[digest] = rel
        t0 = time.time()
        try:
            obj = load_pickle_stub(path)
        except Exception as e:
            entry['error'] = 'stub load failed: %r' % (e,)
            self.rep['counts']['pickles_failed'] += 1
            self.rep['pickles'][rel] = entry
            return
        self.rep['counts']['pickles_loaded'] += 1
        out = new_walk_out()
        walk_seeds(obj, '', out)
        if isinstance(obj, dict):
            if 'seeds' in obj:
                entry['seeds_key'] = _summ(obj['seeds'])
            prov = obj.get('provenance')
            if isinstance(prov, dict):
                entry['provenance_seeds'] = _summ(prov.get('seeds'))
                entry['provenance_argv'] = str(prov.get('argv'))[:200]
        del obj
        gc.collect()
        self.rep['errors'].extend(_walk_truncation_errors(out, rel))
        ints = _ints_of_walk(out)
        self.seeds_data.update(i for i in ints if i <= 100000)
        entry['seed_values'] = _fmt_ranges(i for i in ints if i <= 100000)
        entry['intkeyed_paths'] = _intkeyed_inventory(out)
        entry['hits'] = self.in_window(ints)
        entry['load_s'] = round(time.time() - t0, 1)
        self.rep['pickles'][rel] = entry
        self.log('pickle', rel, entry['seed_values'], '%.1fs' % entry['load_s'])

    # -- 5. checkpoints
    def scan_pt(self, path):
        rel = self._rel(path)
        if PER_AGENT_PT.search(os.path.basename(path)):
            return                     # the seed is in the name (scanned above)
        if not os.path.isfile(path):
            self.rep['errors'].append('%s: vanished before it was opened' % rel)
            return
        self.rep['counts']['pt'] += 1
        try:
            import torch
        except Exception as e:
            self.rep['errors'].append('torch unavailable (%s); --pt aborted, '
                                      'checkpoints NOT scanned' % e)
            self.do_pt = False
            self.pt_aborted = True
            return

        class _Shim(object):
            Unpickler = StubUnpickler
            load = staticmethod(pickle.load)
            loads = staticmethod(pickle.loads)
            __name__ = 'pickle'

        obj = None
        try:
            obj = torch.load(path, map_location='cpu', pickle_module=_Shim, weights_only=False)
        except Exception:
            try:
                obj = torch.load(path, map_location='cpu', weights_only=False)
            except Exception as e:
                self.rep['counts']['pt_failed'] += 1
                self.rep['pt'][rel] = {'error': repr(e)[:200]}
                return
        out = new_walk_out()
        meta = obj.get('metadata') if isinstance(obj, dict) else None
        walk_seeds({'metadata': meta} if meta is not None else obj, '', out)
        self.rep['errors'].extend(_walk_truncation_errors(out, rel))
        ints = _ints_of_walk(out)
        hits = self.in_window(ints)
        # A checkpoint is RESULT DATA, exactly as a pickle and a JSON export
        # are, so its integers belong to the inventory of 4 and its hits to
        # the verdict (finish()).  Neither happened until 2026-09-23: the
        # entry below was written, printed by markdown() and left out of
        # every verdict field, so a planted seed of the window read CLEAN.
        self.seeds_data.update(i for i in ints if i <= 100000)
        if out['seedkeys'] or hits:
            self.rep['pt'][rel] = {'seedkeys': out['seedkeys'][:20], 'hits': hits}
        del obj
        gc.collect()

    # -- 6. git history
    def scan_history(self, fh, base):
        """Scan a `git log --all -p` stream, attributing every line to the
        file of its `diff --git` header: the history of a planning document
        is planning material, exactly as the file is.  Returns (lines,
        commits)."""
        cur, n, commits = None, 0, 0
        for ln, line in enumerate(fh, 1):
            n = ln
            line = line.rstrip('\n')
            m = RE_DIFF_HDR.match(line)
            if m:
                cur = m.group(2) if m.group(2) != '/dev/null' else m.group(1)
            elif RE_COMMIT_HDR.match(line):
                cur = None                      # commit message: no file
                commits += 1
            rel = '%s:%s' % (base, cur) if cur else base
            self.scan_line(rel, ln, line, planning=bool(cur) and self.is_planning(cur))
        return n, commits

    def scan_git(self, root):
        """`git log --all -p` of one root.

        A root with a .git entry MUST yield a successful, non-empty log: an
        empty .git directory (~/MARL_code) or a `git archive` export used to
        produce no hits, no error and no trace, so "no seed was ever
        committed" rested on nothing.  Failure is now a hard error."""
        dot = os.path.join(root, '.git')     # a directory, or a worktree file
        if not os.path.exists(dot):
            self.rep['git_roots'].append(
                {'root': root, 'scanned': False, 'reason': 'no .git entry',
                 'lines': 0, 'commits': 0, 'returncode': None})
            return
        rel = 'git:' + os.path.basename(root)
        before = len(self.rep['text_hits'])
        try:
            p = subprocess.Popen(['git', '-C', root, 'log', '--all', '-p', '--no-color'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            self.rep['errors'].append('git log %s: %s' % (root, e))
            self.rep['git_roots'].append(
                {'root': root, 'scanned': False, 'reason': repr(e)[:200],
                 'lines': 0, 'commits': 0, 'returncode': None})
            return
        n = commits = 0
        with io.TextIOWrapper(p.stdout, encoding='utf-8', errors='replace') as fh:
            n, commits = self.scan_history(fh, rel)
        err = p.stderr.read().decode('utf-8', 'replace').strip()
        p.stderr.close()
        rc = p.wait()
        self.rep['git_hits'].extend(self.rep['text_hits'][before:])
        declared = root in self.git_no_repo
        entry = {'root': root, 'scanned': rc == 0 and n > 0, 'lines': n,
                 'commits': commits, 'returncode': rc, 'stderr': err[:400],
                 'declared_not_a_repo': declared}
        self.rep['git_roots'].append(entry)
        if rc != 0 and declared:
            self.rep['coverage_warnings'].append(
                'git log failed in %s (rc=%d) and the root is declared --git-no-repo: '
                'its history is NOT auditable (%s)' % (root, rc, err[:120] or 'no stderr'))
        elif rc != 0:
            self.rep['errors'].append('git log failed in %s (rc=%d): %s'
                                      % (root, rc, err[:200] or '(no stderr)'))
        elif n == 0:
            self.rep['errors'].append('git log produced no output in %s: the .git entry '
                                      'exists but no commit was scanned' % root)
        else:
            self.rep['counts']['git_roots_scanned'] += 1

    def scan_git_log(self, path):
        """Scan a captured `git log --all -p` dump (--git-log).

        The trackB worktree's history lives on the laptop, while every scan
        runs on the server; the dump makes that history auditable there with
        a recorded, reproducible command.  Its sha256 and size are part of
        the report, and an empty or unreadable dump is a hard error."""
        rel = 'gitlog:' + os.path.basename(path)
        entry = {'path': path, 'scanned': False, 'lines': 0, 'commits': 0,
                 'bytes': 0, 'sha256': None}
        if not os.path.isfile(path):
            self.rep['errors'].append('--git-log %s: not a file' % path)
            self.rep['git_logs'].append(entry)
            return
        h = hashlib.sha256()
        try:
            entry['bytes'] = os.path.getsize(path)
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
        except OSError as e:
            self.rep['errors'].append('--git-log %s: %s' % (path, e))
            self.rep['git_logs'].append(entry)
            return
        entry['sha256'] = h.hexdigest()
        before = len(self.rep['text_hits'])
        n = commits = 0
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                n, commits = self.scan_history(fh, rel)
        except OSError as e:
            self.rep['errors'].append('--git-log %s: %s' % (path, e))
            self.rep['git_logs'].append(entry)
            return
        self.rep['git_hits'].extend(self.rep['text_hits'][before:])
        entry.update(scanned=n > 0, lines=n, commits=commits)
        self.rep['counts']['git_log_lines'] += n
        self.rep['git_logs'].append(entry)
        if n == 0:
            self.rep['errors'].append('--git-log %s: empty dump, no history scanned' % path)

    # -- summary
    def finish(self, t0):
        r = self.rep
        r['finished'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        r['elapsed_s'] = round(time.time() - t0, 1)
        # RESULT DATA IS ALL THREE CONTAINER SOURCES.  scan_pt has recorded
        # pt[<file>]['hits'] since checkpoints were added, and nothing read
        # it: a seed of the window in a checkpoint's metadata was printed in
        # the report, printed by markdown(), and then dropped on the way to
        # the verdict, which said "CLEAN - no seed in <lo>-<hi> in any
        # result, name, script, log or history" and exited 0.  An entry that
        # failed to load carries 'error' and no 'hits', hence .get().
        data_hits = sorted(set(
            v for e in r['pickles'].values() for v in (e['hits'] or [])) | set(
            v for e in r['json'].values() if not e['planning'] for v in e['hits']) | set(
            v for e in r['pt'].values() for v in (e.get('hits') or [])))
        text_hits = sorted(set(v for e in r['text_hits'] for v in e['values']))
        name_hits = sorted(set(v for e in r['filename_hits'] if e['seedlike'] and not e['planning']
                               for v in e['tokens']))
        r['all_seeds'] = {
            'data (pickle keys, JSON seed fields and int keys, checkpoint seed metadata, <= 100000)': _fmt_ranges(self.seeds_data),
            'text (values named next to "seed", CLI specs, range(), <= 1000; planning lines excluded)': _fmt_ranges(
                s for s in self.seeds_text if s <= 1000),
            'text (values > 1000: scenario/event and torch seeds)': _fmt_ranges(
                s for s in self.seeds_text if s > 1000),
            'file names (seed<N>, _s<N>, _<a>_<b>.ext)': _fmt_ranges(self.seeds_names),
        }
        r['text_value_sources'] = {str(k): v for k, v in sorted(r['text_value_sources'].items())}
        cov, warn = self.coverage(t0)
        no_hits = not (data_hits or text_hits or name_hits)
        r['verdict'] = {
            'window_hits_in_result_data': data_hits,
            'window_hits_in_text_non_planning': text_hits,
            'window_hits_in_file_names': name_hits,
            'window_hits_in_git_history': sorted(set(v for e in r['git_hits'] for v in e['values'])),
            'window_hits_classified_false_positive': sorted(set(
                v for e in r['false_positive_hits'] for v in e['values'])),
            'no_window_hits': no_hits,
            'coverage_ok': not cov,
            'coverage_failures': cov,
            'coverage_warnings': warn,
            # BOTH halves: an audit that scanned nothing, lost a root, met an
            # error or failed to open a pickle or a checkpoint is INCOMPLETE
            # and must not pass the freeze gate, whatever its hit lists say
            'clean': no_hits and not cov,
        }
        if self.gate_result_data:
            # WHICH files hold the result-data hits, so that a stop names
            # them.  The gate passes only on full coverage AND no hit in any
            # pickle, non-planning JSON or checkpoint.
            files = sorted(
                [k for k, e in r['pickles'].items() if e['hits']]
                + [k for k, e in r['json'].items() if not e['planning'] and e['hits']]
                + [k for k, e in r['pt'].items() if e.get('hits')])
            r['verdict']['result_data_gate'] = {
                'window': [self.lo, self.hi],
                'files_with_result_data_hits': files,
                'planning_json_with_window_seeds': sorted(
                    k for k, e in r['json'].items() if e['planning'] and e['hits']),
                'pass': (not data_hits) and (not files) and (not cov),
            }

    def coverage(self, t0=None):
        """(failures, warnings): why this run may not be used as evidence."""
        r, c = self.rep, self.rep['counts']
        cov, warn = [], list(r.get('coverage_warnings') or [])
        # named explicitly, and not only through `errors`, so that the freeze
        # gate's "every per-root floor was applied" is a field of the report
        if self.min_root_file and not (r.get('min_root_file') or {}).get('ok'):
            cov.append('--min-root-file %s: missing, unreadable, unparsable or empty; '
                       'the per-root floors were NOT applied' % self.min_root_file)
        for root in self.roots:
            if root not in r['per_root']:
                cov.append('root not walked (missing, unreadable or renamed): %s' % root)
            elif r['per_root'][root] == 0:
                # a warning here let a tree that had been moved away, or a path
                # that was not mounted, pass with exit 0 whenever the other
                # roots still met the aggregate floors -- the reviewer reading
                # a '~' line in the summary was the only check left
                cov.append('root walked but EMPTY (0 files): %s' % root)
            elif self.min_per_root and r['per_root'][root] < self.min_per_root:
                cov.append('per-root coverage floor not met: %s %d < %d (--min-per-root)'
                           % (root, r['per_root'][root], self.min_per_root))
        by_base = {}
        for root, n in r['per_root'].items():
            by_base[os.path.basename(root)] = n
        for name in sorted(self.min_root):
            need = self.min_root[name]
            have = by_base.get(name)
            if have is None:
                cov.append('--min-root %s=%d: no scanned root is named %s' % (name, need, name))
            elif have < need:
                cov.append('per-root coverage floor not met: %s %d < %d (--min-root)'
                           % (name, have, need))
        for e in r.get('symlinks_outside_roots') or []:
            warn.append('symbolic link not followed, target %s: %s -> %s'
                        % ('missing' if e['broken'] else 'outside every scanned root',
                           e['path'], e['target']))
        if r['home_marl_unscanned']:
            cov.append('directories matching $HOME/MARL* that are not among --roots: %s'
                       % ', '.join(os.path.basename(d) for d in r['home_marl_unscanned']))
        if r['errors']:
            cov.append('%d error(s) recorded (see "errors"); the first is: %s'
                       % (len(r['errors']), r['errors'][0][:160]))
        if c['pickles_failed']:
            cov.append('%d pickle(s) failed to load' % c['pickles_failed'])
        if c['pt_failed']:
            cov.append('%d checkpoint(s) failed to load' % c['pt_failed'])
        if self.pt_aborted or c['pt_skipped']:
            cov.append('--pt aborted: %d checkpoint(s) never opened' % c['pt_skipped'])
        if self.do_git and not c['git_roots_scanned'] and not self.git_logs:
            cov.append('--git requested but no root produced a git history; '
                       'no commit was scanned')
        for e in r['git_logs']:
            if not e['scanned']:
                cov.append('--git-log %s scanned nothing' % e['path'])
        if r['long_values']:
            cov.append('%d value string(s) hit the %d-character parse limit'
                       % (len(r['long_values']), VALUE_CHARS))
        got = {'files': c['files'], 'text': c['text'],
               'json': c['json_parsed'] + c['json_scanned'],
               'pickles': c['pickles'], 'pt': c['pt'],
               'json_parsed': c['json_parsed'], 'json_scanned': c['json_scanned']}
        for k in sorted(self.min_counts):
            need = int(self.min_counts[k])
            have = got.get(k)
            if have is None:
                cov.append('unknown --min-%s' % k)
            elif have < need:
                cov.append('coverage floor not met: %s %d < %d' % (k, have, need))
        r['coverage_warnings'] = warn
        return cov, warn

    def markdown(self):
        r = self.rep
        c = r['counts']
        L = []
        L.append('Seed audit %d-%d on %s, %s -> %s (%.0f s)' % (
            self.lo, self.hi, r['host'], r['started'], r['finished'], r['elapsed_s']))
        L.append('files %d | text scanned %d | JSON parsed %d + regex-scanned %d | pickles %d '
                 '(loaded %d, duplicates %d, failed %d) | checkpoints %d (failed %d, skipped %d) '
                 '| binary skipped %d'
                 % (c['files'], c['text'], c['json_parsed'], c['json_scanned'], c['pickles'],
                    c['pickles_loaded'], c['pickles_duplicate'], c['pickles_failed'], c['pt'],
                    c['pt_failed'], c['pt_skipped'], c['skipped_binary']))
        L.append('')
        L.append('Per root: ' + ', '.join('%s=%d' % (os.path.basename(k), v) for k, v in r['per_root'].items()))
        mrf = r.get('min_root_file')
        if mrf:
            L.append('Per-root floors from %s: %d parsed, sha256 %s (%s)'
                     % (mrf['path'], len(mrf['floors']), (mrf['sha256'] or '-')[:16],
                        'applied' if mrf['ok'] else 'NOT APPLIED'))
        sh = r.get('synced_head')
        if sh:
            L.append('Tree .synced_head: %s (%s, %s)'
                     % (sh.get('value') or '-', sh.get('mtime_utc') or '-',
                        sh.get('error') or 'read'))
        L.append('Skipped directory names: ' + ', '.join(r.get('skip_dirs') or sorted(SKIP_DIRS)))
        L.append('Symbolic links met (never followed): %d (%d directories); '
                 'targets outside every scanned root or broken: %d'
                 % (c.get('symlinks', 0), c.get('symlink_dirs', 0),
                    len(r.get('symlinks_outside_roots') or [])))
        if r['home_marl_dirs']:
            L.append('$HOME/MARL* directories: %d (%s)' % (
                len(r['home_marl_dirs']),
                ', '.join(os.path.basename(d) for d in r['home_marl_dirs'])))
        gr = [e for e in r['git_roots'] if e['scanned']]
        if r['git_roots'] or r['git_logs']:
            L.append('Git history: %d root(s) scanned of %d with a .git entry; dumps: %s'
                     % (len(gr), len([e for e in r['git_roots'] if e['returncode'] is not None]),
                        ', '.join('%s (%d commits, %d lines, sha256 %s)'
                                  % (os.path.basename(e['path']), e['commits'], e['lines'],
                                     (e['sha256'] or '-')[:16]) for e in r['git_logs']) or 'none'))
        L.append('')
        v = r['verdict']
        if not v['coverage_ok']:
            head = 'INCOMPLETE - coverage refused, this run is not evidence'
        elif v['no_window_hits']:
            head = ('CLEAN - no seed in %d-%d in any result, name, script, log or history'
                    % (self.lo, self.hi))
        else:
            head = 'TOUCHED - see hits'
        L.append('VERDICT: %s' % head)
        L.append('  result data hits   : %s' % (v['window_hits_in_result_data'] or 'none'))
        L.append('  text hits (non-plan): %s' % (v['window_hits_in_text_non_planning'] or 'none'))
        L.append('  file-name hits     : %s' % (v['window_hits_in_file_names'] or 'none'))
        L.append('  git-history hits   : %s' % (v['window_hits_in_git_history'] or 'none'))
        L.append('  classified false positives (--false-positive): %s' % (
            v['window_hits_classified_false_positive'] or 'none'))
        g = v.get('result_data_gate')
        if g:
            L.append('  RESULT-DATA GATE   : %s (--gate-result-data: zero seeds of %d-%d in '
                     'any pickle, non-planning JSON or checkpoint, on full coverage)'
                     % ('PASS' if g['pass'] else 'FAIL', self.lo, self.hi))
            for f in g['files_with_result_data_hits'][:40]:
                L.append('    ! result data: %s' % f)
            for f in g['planning_json_with_window_seeds'][:40]:
                L.append('    ~ planning JSON naming the window (classified by a marker): %s' % f)
        if self.planning_markers:
            L.append('  planning markers of this audit (--planning-marker): %s'
                     % ', '.join(self.planning_markers))
        L.append('  coverage           : %s' % ('ok' if v['coverage_ok'] else 'REFUSED'))
        for f in v['coverage_failures']:
            L.append('    ! %s' % f)
        for w in v['coverage_warnings']:
            L.append('    ~ %s' % w)
        L.append('')
        L.append('Seeds ever used / referenced:')
        for k, val in r['all_seeds'].items():
            L.append('  %-70s %s' % (k, val))
        src = r['text_value_sources']
        odd = [k for k in src if int(k) not in self.seeds_data]
        if odd:
            L.append('  text values not present in any result data, first source of each:')
            for k in odd[:40]:
                L.append('    %5s  %s' % (k, src[k]))
        L.append('')
        if r['text_hits']:
            L.append('Text hits (non-planning), first 60:')
            for e in r['text_hits'][:60]:
                L.append('  %s:%d [%s] %s | %s' % (e['path'], e['line'], ','.join(e['kinds']),
                                                    e['values'], e['snippet'][:110]))
        if r['false_positive_hits']:
            L.append('Hits classified as false positives by --false-positive (%d lines):' % len(r['false_positive_hits']))
            for e in r['false_positive_hits'][:20]:
                L.append('  %s:%d %s | %s' % (e['path'], e['line'], e['values'], e['snippet'][:110]))
        rb = r.get('range_nonseed_benign') or []
        if rb:
            byvar = {}
            for e in rb:
                byvar[e['var']] = byvar.get(e['var'], 0) + 1
            L.append('range() loops over a tick/episode variable on lines without "seed" '
                     '(%d lines, listed, not counted; loop variables %s):'
                     % (len(rb), json.dumps(byvar, sort_keys=True)))
            for e in rb[:20]:
                L.append('  %s:%d [%s] %s | %s' % (e['path'], e['line'], e['var'], e['values'], e['snippet'][:110]))
        if r['planning_hits']:
            L.append('Planning-document mentions (%d lines): %s' % (
                len(r['planning_hits']), ', '.join(sorted(set(e['path'] for e in r['planning_hits'])))))
        if r['filename_hits']:
            L.append('File names with a token in the window (%d):' % len(r['filename_hits']))
            for e in r['filename_hits'][:40]:
                L.append('  %s  tokens=%s seedlike=%s planning=%s' % (
                    e['path'], e['tokens'], e['seedlike'], e['planning']))
        h = r['broad_tokens']['histogram']
        L.append('Other %d-%d tokens on "seed" lines, by left context: %s' % (
            self.lo, self.hi, json.dumps(h, sort_keys=True)))
        oth = r['broad_tokens']['other']
        if oth:
            L.append('  unclassified (%d shown):' % len(oth))
            for e in oth[:40]:
                L.append('    %s:%d ctx=%r tok=%d | %s' % (e['path'], e['line'], e['ctx'], e['token'],
                                                          e['snippet'][:100]))
        pk = r['pickles']
        L.append('Pickles (%d): seed values per file (int-keyed paths in the JSON report)' % len(pk))
        for k in sorted(pk):
            e = pk[k]
            tag = ('dup of %s' % e['duplicate_of']) if e['duplicate_of'] else (
                e['error'] or ('%.1fs' % e.get('load_s', 0)))
            L.append('  %-70s %-28s seeds_key=%s %s' % (k, e['seed_values'], e['seeds_key'], tag))
        fb = sorted(k for k, e in r['json'].items() if e.get('scan', 'parsed') != 'parsed')
        L.append('JSON files NOT parsed (above --json-max or unparseable), scanned by the '
                 'fallback path (%d): %s' % (len(fb), ', '.join(fb) or 'none'))
        js = {k: e for k, e in r['json'].items() if e['n_intkeyed'] or e['n_seedkeys']}
        L.append('JSON files with seed content (%d of %d):' % (len(js), len(r['json'])))
        for k in sorted(js):
            e = js[k]
            L.append('  %-70s %s%s%s' % (
                k, e['seed_values'],
                '  [%s]' % e['scan'] if e.get('scan', 'parsed') != 'parsed' else '',
                '  HIT %s' % e['hits'] if e['hits'] else ''))
            for line in (e.get('intkeyed_paths') or [])[:6]:
                if line.split(':')[0].strip() not in ('/instances', ''):
                    L.append('      %s' % line[:150])
        if r['pt']:
            L.append('Checkpoints with seed metadata or errors (%d):' % len(r['pt']))
            for k in sorted(r['pt'])[:40]:
                L.append('  %s: %s' % (k, json.dumps(r['pt'][k])[:160]))
        so = r.get('symlinks_outside_roots') or []
        if so:
            L.append('Symbolic links whose target is outside every scanned root, '
                     'or broken (%d):' % len(so))
            for e in so[:40]:
                L.append('  %s -> %s%s' % (e['path'], e['target'],
                                           ' (broken)' if e['broken'] else ''))
        if r['errors']:
            L.append('Errors (%d):' % len(r['errors']))
            for e in r['errors'][:30]:
                L.append('  ' + e)
        return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lo', type=int, required=True)
    ap.add_argument('--hi', type=int, required=True)
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--out', required=True, help='JSON report path (outside the scanned roots)')
    ap.add_argument('--json-max', type=int, default=64 * 1024 * 1024)
    ap.add_argument('--pt', action='store_true', help='also open torch checkpoints')
    ap.add_argument('--git', action='store_true', help='also scan git history of each root')
    ap.add_argument('--git-log', action='append', default=[], metavar='FILE',
                    help='scan a captured `git log --all -p` dump (a repository on '
                         'another host, e.g. the trackB worktree); repeatable')
    ap.add_argument('--git-no-repo', action='append', default=[], metavar='ROOT',
                    help='a root whose .git entry is known not to be a repository (an '
                         'empty .git directory left behind); the failure is recorded as '
                         'a warning naming the root instead of refusing the verdict')
    ap.add_argument('--no-home-check', action='store_true',
                    help='do not require every $HOME/MARL* directory to be among '
                         '--roots (ad-hoc single-tree audits only; never at a freeze)')
    ap.add_argument('--min-files', type=int, default=0, metavar='N')
    ap.add_argument('--min-text', type=int, default=0, metavar='N')
    ap.add_argument('--min-json', type=int, default=0, metavar='N')
    ap.add_argument('--min-pickles', type=int, default=0, metavar='N')
    ap.add_argument('--min-pt', type=int, default=0, metavar='N',
                    help='coverage floors: the verdict is INCOMPLETE (exit 4) when the '
                         'run scanned fewer files / text files / JSON files / pickles / '
                         'checkpoints than the recorded audit did')
    ap.add_argument('--min-per-root', type=int, default=0, metavar='N',
                    help='every root must hold at least N files (the aggregate floors '
                         'above cannot see one root that lost its content)')
    ap.add_argument('--min-root-file', default=None, metavar='PATH',
                    help='file of per-root floors, one NAME=N per line (# comments, '
                         'blank lines ignored).  The tool reads it itself and records '
                         'its path, sha256 and every parsed floor in the report; '
                         'missing, unreadable, unparsable or empty is a COVERAGE '
                         'FAILURE, not a silently floor-less run')
    ap.add_argument('--tree', default=None, metavar='PATH',
                    help='the tree this audit is run from (default: cwd); its '
                         '.synced_head is recorded with the verdict')
    ap.add_argument('--min-root', action='append', default=[], metavar='NAME=N',
                    help='a floor for one root, by directory base name, e.g. '
                         'MARL_watch=12 (repeatable)')
    ap.add_argument('--false-positive', action='append', default=[], metavar='SUBSTR',
                    help='a line snippet substring classified by a human as not a seed use; '
                         'matching hits go to their own bucket (repeatable)')
    ap.add_argument('--skip-dir', action='append', default=[], metavar='NAME',
                    help='a directory name not to walk, in addition to %s (repeatable)'
                         % ', '.join(sorted(SKIP_DIRS)))
    ap.add_argument('--planning-marker', action='append', default=[], metavar='SUBSTR',
                    help='a path substring classified as PLANNING for this audit, in '
                         'addition to the built-in planning documents (repeatable; '
                         'recorded in the report).  It moves text lines, JSON files '
                         'and file names only: a pickle or a checkpoint is result '
                         'data whatever its path')
    ap.add_argument('--gate-result-data', action='store_true',
                    help='the R3 freeze gate: exit 5 when any seed of the window is '
                         'in RESULT DATA (a pickle, a non-planning JSON file or a '
                         'checkpoint), naming the files; exit 4 (INCOMPLETE) still '
                         'wins, and 0 / 3 keep their meaning for the remaining '
                         'text, file-name and history hits')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args(argv)
    mins = {k: v for k, v in (('files', a.min_files), ('text', a.min_text),
                              ('json', a.min_json), ('pickles', a.min_pickles),
                              ('pt', a.min_pt)) if v}
    per_root = {}
    for spec in a.min_root:
        name, sep, n = spec.partition('=')
        if not sep or not n.strip().isdigit() or not name.strip():
            ap.error('--min-root expects NAME=N, got %r' % spec)
        per_root[name.strip()] = int(n)
    au = Audit(a.lo, a.hi, a.roots, a.json_max, a.pt, a.git, a.verbose,
               false_positives=a.false_positive, skip_dirs=a.skip_dir,
               git_logs=a.git_log, min_counts=mins, home_check=not a.no_home_check,
               git_no_repo=a.git_no_repo, min_per_root=a.min_per_root,
               min_root=per_root, min_root_file=a.min_root_file, tree=a.tree,
               planning_markers=a.planning_marker,
               gate_result_data=a.gate_result_data,
               argv=list(sys.argv[1:] if argv is None else argv))
    au.run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(au.rep, f, indent=1, sort_keys=True, default=str)
    print(au.markdown())
    print('\nwritten', a.out)
    v = au.rep['verdict']
    if not v['coverage_ok']:
        return 4
    if a.gate_result_data and not v['result_data_gate']['pass']:
        return 5
    return 0 if v['clean'] else 3


if __name__ == '__main__':
    sys.exit(main())
