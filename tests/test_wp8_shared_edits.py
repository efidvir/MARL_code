# -*- coding: utf-8 -*-
"""WP8-FIX -- the shared-file declaration is machine-checked, not asserted.

`docs/SHARED_FILE_EDITS.md` is the artefact the four-branch merge is
adjudicated against: one entry per hunk of `git diff -U0 9dec943 HEAD`, for
every source path branch `trackAgent` does not own.  Hand-maintained, it was
found incomplete in THREE consecutive review rounds -- most recently with
four undeclared hunks in `train_on_comparison.py` (one of them in a function
the declaration never named) and a non-owned total that enumerated to 202
while the text said 206.  Each time the report asserted the list had been
"re-derived mechanically".

So the list is generated (`tools/shared_file_edits.py`) and checked here.

TWO CHECKS, and which one runs depends on what the tree carries:

* WITH git history -- the laptop worktree -- the declaration is REGENERATED
  from the diff and compared to the committed file, byte for byte.  That is
  the check that closes the defect.
* WITHOUT it -- the server tree is extracted with `git archive`, so it has
  no `.git` at all -- the committed file is checked against the committed
  hunk table `tests/data/shared_file_edits_bde51a9.tsv` (generated from the
  same diff, and itself byte-compared wherever git exists): the document's
  (path, region, range) set for the non-owned paths must EQUAL the table's,
  and its owned counts must equal the table's, so an OMITTED hunk -- a row
  dropped together with its heading count -- is caught on the server too.
  (Carry-forward defect 5 of the trackAgent review: the earlier fallback
  mapped the DECLARED ranges back onto the AST and could not see an omitted
  one; after the merge the working tree is no longer the branch head
  either, so that mapping is gone.)

The declaration is FROZEN at the trackAgent head the merge consumed
(`GEN.HEAD_REV`); see tools/shared_file_edits.py.

Neither path returns early and neither reports a pass without counting what
it checked; both assert a non-zero amount of work.

    PYTHONHASHSEED=0 python3 tests/run_tests.py wp8_shared
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, 'tools') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'tools'))

import shared_file_edits as GEN      # noqa: E402

DOC = os.path.join(ROOT, 'docs', 'SHARED_FILE_EDITS.md')

_SECTION = re.compile(r'^### `([^`]+)` -- (\d+) hunks?$')
_ROW = re.compile(r'^\| `([^`]+)` \| (\d+) \| ([^|]*) \| (.+) \|$')
_TOTAL = re.compile(r'^\| (?:\*\*)?([^|*]+?)(?:\*\*)? \| (?:\*\*)?(\d+)'
                    r'(?:\*\*)? \| (?:\*\*)?(\d+)(?:\*\*)? \|$')


def _doc():
    with open(DOC, encoding='utf-8') as f:
        return f.read()


def _has_git_history():
    """True when this tree can be diffed against the branch base.

    Asked of GIT, not of the filesystem: in a `git worktree` -- which is what
    this branch is checked out as -- `.git` is a FILE, so an isdir() test
    reports "no history" on the one tree that has it and quietly downgrades
    the byte comparison to the fallback."""
    for rev in (GEN.BASE, GEN.HEAD_REV):
        try:
            out = subprocess.run(
                ('git', 'rev-parse', '--verify', rev + '^{commit}'),
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return False
        if out.returncode != 0:
            return False
    return True


def _data_rows():
    """[(path, region, (start, length))] from the committed hunk table."""
    rows = []
    with open(os.path.join(ROOT, *GEN.DATA.split('/')), 'rb') as f:
        text = f.read().decode('utf-8')
    assert chr(13) not in text, 'the hunk table is not LF'
    for line in text.split(chr(10)):
        if not line or line.startswith('#'):
            continue
        path, region, rng = line.split(chr(9))
        a, b = rng.split(',')
        rows.append((path, region, (int(a), int(b))))
    return rows


def _parse_sections(text):
    """{path: (declared_total, [(region, n, [ranges], why), ...])}."""
    out = {}
    path = None
    for line in text.split(chr(10)):
        m = _SECTION.match(line)
        if m:
            path = m.group(1)
            out[path] = (int(m.group(2)), [])
            continue
        if path is None:
            continue
        m = _ROW.match(line)
        if m and m.group(1) != 'region':
            ranges = []
            for tok in m.group(3).split():
                a, b = tok.split(',')
                ranges.append((int(a), int(b)))
            out[path][1].append((m.group(1), int(m.group(2)), ranges,
                                 m.group(4).strip()))
    return out


def _parse_reconciliation(text):
    rows = {}
    for line in text.split(chr(10)):
        m = _TOTAL.match(line)
        if m:
            rows[m.group(1).strip()] = (int(m.group(2)), int(m.group(3)))
    return rows


def _parse_owned(text):
    """The `| \\`path\\` | n |` rows of the owned table."""
    out = {}
    in_owned = False
    for line in text.split(chr(10)):
        if line.startswith('## Paths this track owns'):
            in_owned = True
            continue
        if in_owned and line.startswith('## '):
            in_owned = False
        if not in_owned:
            continue
        m = re.match(r'^\| `([^`]+)` \| (\d+) \|$', line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def test_the_declaration_exists_and_says_it_is_generated():
    assert os.path.exists(DOC), (
        'docs/SHARED_FILE_EDITS.md is missing: the merge has nothing to be '
        'adjudicated against')
    text = _doc()
    assert 'GENERATED FILE' in text, text[:200]
    assert 'tools/shared_file_edits.py' in text
    assert GEN.BASE in text, 'the declaration does not name its base commit'
    assert len(text) > 4000, len(text)


def test_the_declaration_matches_the_diff_or_the_tree():
    """The defect, closed.  With git history this REGENERATES the document
    from `git diff -U0` and compares; without it, every declared range is
    mapped back onto the AST of the file it names."""
    text = _doc()
    sections = _parse_sections(text)
    n_rows = sum(len(v[1]) for v in sections.values())
    n_hunks = sum(n for v in sections.values() for _r, n, _g, _w in v[1])
    assert n_rows > 20, 'only %d declared regions' % n_rows
    assert n_hunks > 100, 'only %d declared hunks' % n_hunks

    with open(DOC, 'rb') as f:
        assert b'\r' not in f.read(), (
            'docs/SHARED_FILE_EDITS.md has CRLF line endings (carry-forward '
            'defect 3); regenerate it with --write')
    if _has_git_history():
        regenerated = GEN.build()
        assert text == regenerated, (
            'docs/SHARED_FILE_EDITS.md DISAGREES with git diff -U0 %s %s.  '
            'Regenerate it:  python3 tools/shared_file_edits.py --write'
            % (GEN.BASE, GEN.HEAD_REV))
        with open(os.path.join(ROOT, *GEN.DATA.split('/')), 'rb') as f:
            assert f.read().decode('utf-8') == GEN.hunk_table(), (
                '%s DISAGREES with the diff; regenerate with --write'
                % GEN.DATA)
        # and the generator really did read the diff, not an empty one
        src, _other = GEN.changed_paths()
        assert len(src) > 20, len(src)
        print('[wp8_shared] regenerated from git: %d regions, %d hunks, '
              '%d source paths' % (n_rows, n_hunks, len(src)))
    # BOTH trees: the document against the committed hunk table, as SETS,
    # so an omitted hunk (or a row dropped with its heading count) fails.
    data = _data_rows()
    assert len(data) > 250, 'the hunk table holds only %d rows' % len(data)
    declared = set()
    for path, (_total, rows) in sections.items():
        for region, _n, ranges, _why in rows:
            for rng in ranges:
                declared.add((path, region, rng))
    table_shared = set(r for r in data if r[0] in sections)
    table_paths = set(r[0] for r in data)
    owned = _parse_owned(text)
    missing_paths = sorted(p for p in table_paths
                           if p not in sections and p not in owned)
    assert not missing_paths, (
        'source paths in the diff that the document names nowhere: %r'
        % missing_paths)
    omitted = sorted(table_shared - declared)
    invented = sorted(declared - table_shared)
    assert not omitted and not invented, (
        'the declaration and the hunk table disagree: %d omitted %r, %d not '
        'in the diff %r' % (len(omitted), omitted[:5], len(invented),
                            invented[:5]))
    per_owned = {}
    for path, _region, _rng in data:
        if path in owned:
            per_owned[path] = per_owned.get(path, 0) + 1
    assert per_owned == owned, (per_owned, owned)
    checked = len(declared) + sum(owned.values())
    assert checked == len(data) and checked > 250, (checked, len(data))
    if not _has_git_history():
        print('[wp8_shared] no git history here; %d hunks checked against '
              'the committed table %s' % (checked, GEN.DATA))


def test_the_arithmetic_reconciles():
    """D8.21 rule 3, enforced: the per-region counts sum to the per-path
    counts, and those sum to the stated totals.  This is the half that
    failed last round -- 95 owned + 202 enumerated non-owned against a
    stated 206 and a stated whole-tree 301."""
    text = _doc()
    sections = _parse_sections(text)
    owned = _parse_owned(text)
    rec = _parse_reconciliation(text)

    assert sections, 'no per-path sections parsed out of the declaration'
    assert owned, 'no owned-path table parsed out of the declaration'

    # 1. per-region -> per-path
    bad = []
    for path, (declared, rows) in sections.items():
        summed = sum(n for _r, n, _g, _w in rows)
        if summed != declared:
            bad.append('%s: heading says %d, regions sum to %d'
                       % (path, declared, summed))
        for region, n, ranges, why in rows:
            if len(ranges) != n:
                bad.append('%s::%s: count %d, %d ranges listed'
                           % (path, region, n, len(ranges)))
            if len(why) < 20:
                bad.append('%s::%s: no reason recorded' % (path, region))
    assert not bad, bad

    # 2. per-path -> the reconciliation table
    shared_paths = len(sections)
    shared_hunks = sum(v[0] for v in sections.values())
    owned_paths = len(owned)
    owned_hunks = sum(owned.values())

    assert 'NOT owned by this track' in rec, sorted(rec)
    assert rec['NOT owned by this track'] == (shared_paths, shared_hunks), (
        'the declaration says %r for the non-owned set; the entries '
        'enumerate to (%d, %d)'
        % (rec['NOT owned by this track'], shared_paths, shared_hunks))
    assert rec['owned by this track'] == (owned_paths, owned_hunks), (
        rec['owned by this track'], (owned_paths, owned_hunks))

    whole = [k for k in rec if k.startswith('all source paths')]
    assert len(whole) == 1, sorted(rec)
    assert rec[whole[0]] == (owned_paths + shared_paths,
                             owned_hunks + shared_hunks), (
        'the whole-tree row %r does not equal owned + non-owned = (%d, %d)'
        % (rec[whole[0]], owned_paths + shared_paths,
           owned_hunks + shared_hunks))

    # 3. the arithmetic is WRITTEN OUT, so a reader can check it
    sentence = ('`%d + %d = %d` hunks over `%d + %d = %d` source paths.'
                % (owned_hunks, shared_hunks, owned_hunks + shared_hunks,
                   owned_paths, shared_paths, owned_paths + shared_paths))
    assert sentence in text, (
        'the declaration does not state the reconciliation as an arithmetic: '
        'expected %r' % sentence)

    assert shared_hunks > 100 and owned_hunks > 50, (shared_hunks, owned_hunks)


def test_no_forbidden_path_is_touched():
    """The paths this track may not edit carry zero hunks, and the document
    says so path by path rather than claiming it in prose."""
    text = _doc()
    assert '## Forbidden paths' in text
    rows = dict((m.group(1), int(m.group(2))) for m in
                (re.match(r'^\| `([^`]+)` \| (\d+) \|$', ln)
                 for ln in text.split(chr(10))) if m)
    missing = [p for p in GEN.FORBIDDEN if p not in rows]
    assert not missing, (
        'the declaration does not report these forbidden paths: %r' % missing)
    touched = dict((p, rows[p]) for p in GEN.FORBIDDEN if rows[p] != 0)
    assert not touched, (
        'a FORBIDDEN path carries hunks in this branch: %r' % touched)
    assert len(GEN.FORBIDDEN) >= 8, GEN.FORBIDDEN


def test_the_undeclared_regions_of_the_previous_round_are_declared_now():
    """The four hunks the last review found missing, by NAME.  A generated
    list is only an improvement if it actually contains them.  The names are
    pinned; the line ranges are not, because they move whenever the file
    legitimately changes and the byte comparison above already covers
    them."""
    sections = _parse_sections(_doc())
    assert 'train_on_comparison.py' in sections, sorted(sections)
    total, rows = sections['train_on_comparison.py']
    by_region = dict((r, n) for r, n, _g, _w in rows)

    # the three regions the previous declaration omitted entirely
    assert by_region.get('run_comparison_worker_episode._act', 0) >= 2, (
        'the WP6 critic-feature switch and the WP7 head-set store are still '
        'not declared: %r' % sorted(by_region))
    assert by_region.get('all_pinnable_heads', 0) >= 1, sorted(by_region)
    assert by_region.get('run_eval_episode', 0) >= 1, (
        'run_eval_episode is STILL absent -- the previous declaration never '
        'named that function at all: %r' % sorted(by_region))

    # and the regions the previous declaration did carry are still there
    for region in ('main', 'run_comparison_worker_episode',
                   'run_comparison_worker_episode._complete_pending',
                   'resolve_frozen_run_shape'):
        assert region in by_region, (region, sorted(by_region))
    assert total >= 22, (
        'train_on_comparison.py carried 22 hunks when this was written and '
        'the declaration now says %d' % total)


def test_the_generator_refuses_a_region_with_no_reason():
    """The prose half cannot go stale while the counts stay right: a region
    with no entry in REASONS/FILE_REASONS fails the generator, and an entry
    for a region that has left the diff fails it too.

    The resolution rule is exercised here on every tree.  Where the diff can
    be read -- the laptop worktree -- the two REFUSALS are exercised as
    well, by building with the tables perturbed."""
    # 1. the resolution rule itself: no git needed
    key = sorted(GEN.REASONS)[0]
    path, region = key.split('::')
    assert GEN.reason_for(path, region) == GEN.REASONS[key]
    assert GEN.reason_for('sixg_sim/nowhere.py', 'not_a_region') is None, (
        'the generator invents a reason for a region it has never been '
        'told about')
    file_key = sorted(GEN.FILE_REASONS)[0]
    assert GEN.reason_for(file_key, 'any_region') == GEN.FILE_REASONS[file_key]
    assert len(GEN.REASONS) >= 20 and len(GEN.FILE_REASONS) >= 10, (
        len(GEN.REASONS), len(GEN.FILE_REASONS))

    # 2. the refusals, where the diff is readable
    probes = []
    if _has_git_history():
        saved_r, saved_f = dict(GEN.REASONS), dict(GEN.FILE_REASONS)
        try:
            del GEN.REASONS[key]
            GEN.FILE_REASONS.pop(path, None)
            try:
                GEN.build()
            except SystemExit as exc:
                probes.append(('missing', str(exc)))

            GEN.REASONS, GEN.FILE_REASONS = dict(saved_r), dict(saved_f)
            GEN.REASONS['sixg_sim/nowhere.py::not_a_region'] = 'x' * 30
            try:
                GEN.build()
            except SystemExit as exc:
                probes.append(('stale', str(exc)))
        finally:
            GEN.REASONS, GEN.FILE_REASONS = saved_r, saved_f

        assert [k for k, _ in probes] == ['missing', 'stale'], (
            'the generator accepted a region with no reason, or a reason for '
            'a region that is not in the diff: %r' % (probes,))
        assert 'no reason recorded' in probes[0][1], probes[0][1]
        assert 'no longer in the diff' in probes[1][1], probes[1][1]
        # the tables went back, and the committed document still matches
        assert GEN.build() == _doc()
        print('[wp8_shared] both generator refusals exercised')
    else:
        print('[wp8_shared] no git history here; the two build() refusals '
              'were not exercised, the resolution rule was')


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
