# -*- coding: utf-8 -*-
"""Derive the shared-file-edit declaration for branch `trackAgent`.

WHY THIS IS A PROGRAM.  The declaration is the artefact the four-branch
merge is adjudicated against: one entry per hunk of `git diff -U0 <base>
HEAD`, for every source path this track does not own, grouped by the region
of the file it lands in.  Hand-maintained, it was found incomplete in three
consecutive review rounds -- the last time with four undeclared hunks in
`train_on_comparison.py` and a non-owned total that enumerated to 202 while
the text said 206 (D8.26).  So it is no longer written by hand: it is
GENERATED from the diff and the AST, `docs/SHARED_FILE_EDITS.md` is the
generated file, and `tests/test_wp8_shared_edits.py` regenerates it and
fails when the committed document and the diff disagree.

    python3 tools/shared_file_edits.py --write
    python3 tools/shared_file_edits.py --check docs/SHARED_FILE_EDITS.md

FROZEN AT THE BRANCH HEAD (trackAgent merge).  The diff is `BASE..HEAD_REV`,
the trackAgent head the merge into master consumed, and every region is
resolved in the file AT THAT REVISION (`git show`), not in the working tree:
after the merge `HEAD` also carries master's engine work, which this track
did not make.  `--write` also writes DATA, the same hunks as a table (every
source path, owned or not), so a tree WITHOUT git -- the server, which is a
`git archive` -- can still prove the document omits no hunk.  Both files are
written in binary with LF endings: the document was once committed with
CRLF because a Windows shell redirected text-mode stdout into it.

SCOPE.  Source paths only (`*.py`).  A declaration that also counted
documentation could not be committed: this file and `docs/DECISIONS.md` are
themselves in the diff, so writing the total down would change it.  The
non-source paths in the diff are listed by NAME at the end, without counts,
which is stable.  Nothing in the merge turns on a hunk count for a
Markdown file.

REASONS are the one hand-written part.  They are keyed by (path, region),
and the generator fails if a region has no reason or a reason names a region
that is no longer in the diff -- so the prose cannot go stale silently while
the counts stay right.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

BASE = '9dec943'
# The branch head the merge consumed.  The declaration is a FROZEN record of
# what branch trackAgent changed between BASE and this commit (see the module
# docstring); pinned, it regenerates identically for as long as it exists.
HEAD_REV = 'bde51a9'
# The hunk table a tree without git checks the document against.
DATA = 'tests/data/shared_file_edits_bde51a9.tsv'
DOC = 'docs/SHARED_FILE_EDITS.md'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── What this track owns, spelled out ────────────────────────────────────
# The work-package grant, verbatim: `sixg_sim/agent.py` and the policy/critic
# network, the reward function, the observation and action schema code,
# `make_untrained_checkpoints.py`, and this track's own WP6/WP7/WP8 tests.
# ANYTHING NOT LISTED HERE IS DECLARED, whether or not this task touched it.
# `train_on_comparison.py` is deliberately absent: the grant there is narrow
# ("the action/reward handling inside"), so every one of its hunks is
# declared and marked INSIDE or OUTSIDE that scope.
OWNED = (
    'sixg_sim/agent.py',
    'sixg_sim/reward_spec.py',
    'make_untrained_checkpoints.py',
)
OWNED_PREFIXES = (
    'tests/test_wp6_',
    'tests/test_wp7_',
    'tests/test_wp8_',
)

# Paths this track may not edit at all (RERUN_SPEC / the WP grant).  Each is
# reported with its hunk count, which must be 0.
FORBIDDEN = (
    'sixg_sim/topology.py',
    'sixg_sim/link_state_detector.py',
    'sixg_sim/link_state_service.py',
    'sixg_sim/tick_step.py',
    'sixg_sim/transport_relay_model.py',
    'confirmatory_analysis.py',
    'extract_timelines.py',
    'dump_results_json.py',
)

# ── The hand-written half: why each region was touched ───────────────────
# Keyed 'path::region'.  A region without an entry fails the generator; an
# entry for a region no longer in the diff fails it too.
REASONS = {
    'sixg_sim/simulation.py::<module>':
        "WP7: the relay-head index rule and the postcard heartbeat "
        "constant are imported from `agent.py`, so the action space has "
        "ONE definition; the import of a deleted helper goes.",
    'sixg_sim/simulation.py::Simulator':
        "Class body, REWARD region. The reward-rate constants were "
        "DELETED (they are fields of reward_spec.RewardCoefficients), and "
        "with D8.23 the reunification cadence went with them -- the "
        "published field is read instead of a second copy. Includes the "
        "reachability-constants block. INSIDE this track's grant (the "
        "reward function).",
    'sixg_sim/simulation.py::Simulator.__init__':
        "WP7: the postcard heartbeat is unconditional and hard-wired "
        "because its action head is deleted; the old switch survives only "
        "as a tripwire that refuses a stale setting.",
    'sixg_sim/simulation.py::Simulator.compute_global_connectivity_reward':
        "The team objective itself: it MEASURES and DELEGATES to "
        "`reward_spec.global_reward`, and reads the published cadence "
        "(D8.23). INSIDE this track's grant (the reward function).",
    'sixg_sim/simulation.py::Simulator._update_phy_mac_observations':
        "WP6 observation schema: the restored-user delta the reach term "
        "prices is computed against the relay-free graph here. INSIDE "
        "this track's grant (observation schema code).",
    'sixg_sim/simulation.py::Simulator._step_phy_mac':
        "WP7: the scheduler and handover WRITES are deleted with their "
        "heads; nothing ever read either field. Action-application "
        "region, NARROW EXCEPTION, kept minimal.",
    'sixg_sim/simulation.py::Simulator._compute_connectivity_state':
        "WP6 actor field `c_live_infra_neighbours`, with the WP3 seam "
        "recorded in the comment (D6.4): it stays the true-graph count "
        "until the engine track lands the link-state service. INSIDE this "
        "track's grant (observation schema code).",
    'sixg_sim/simulation.py::Simulator._build_neighbor_radio_summary':
        "WP6 observation schema: the neighbour radio summary the actor "
        "reads. INSIDE this track's grant.",
    'sixg_sim/simulation.py::Simulator._build_agent_observations':
        "WP6 observation schema: the observation vector is built to the "
        "new field list. INSIDE this track's grant.",
    'sixg_sim/simulation.py::Simulator._build_agent_observations._build_one_obs':
        "WP6 observation schema: the per-agent vector. INSIDE this "
        "track's grant.",
    'sixg_sim/simulation.py::Simulator._relay_hysteresis':
        "WP7: both sides of the hysteresis comparison are HEAD indices "
        "now; the stored enum index would have indexed a 3-wide logit row "
        "with a value up to 4. Action-application region, NARROW "
        "EXCEPTION.",
    'sixg_sim/simulation.py::Simulator._execute_agent_actions':
        "WP7 ACTION APPLICATION: the deleted heads are no longer applied "
        "and the surviving ones are read at their new indices. This is "
        "the NARROW EXCEPTION region the work package grants, and the "
        "largest single merge surface on this branch.",
    'run_timeline_comparison.py::<module>':
        "WP6/WP7: the evaluation driver imports the new critic network, "
        "observation widths and the schema gate.",
    'run_timeline_comparison.py::_establish_relay_links':
        "WP7: the handover and scheduler arguments are gone with their "
        "heads; the SDN arm is bit-identical without them because the "
        "scheduler write only restored the value it had just read.",
    'run_timeline_comparison.py::_load_into_agent':
        "WP6: the observation-dimension ADAPTATION is removed -- a "
        "checkpoint from another schema is refused by the strict gate "
        "before this is reached, so the function always returns False.",
    'run_timeline_comparison.py::run_physics_pass':
        "WP6: the strict checkpoint gate runs BEFORE anything is read out "
        "of a checkpoint, and the fc1 zero-padding that used to make a "
        "narrower checkpoint load is gone (the WP6 schema drops INTERIOR "
        "columns, so padding would mis-align every column after the first "
        "drop). Checkpoint-loading region; the physics pass proper is "
        "untouched.",
    'run_timeline_comparison.py::run_physics_pass._act_fn':
        "WP6/WP7 PPO COLLECTION: the critic feature vector (26 actor "
        "fields followed by 15 global ones), and the relay HEAD index "
        "rather than the enum index, which is what PPO scores. "
        "Action/reward handling.",
    'train_on_comparison.py::all_pinnable_heads':
        "WP7: `ALL_PINNABLE_HEADS` was a second copy of the head list; it "
        "is derived from `DISCRETE_HEADS` now. INSIDE the action-handling "
        "grant. UNDECLARED until D8.26.",
    'train_on_comparison.py::main':
        "WP6/WP7/WP8: the flags, banner and run summary follow the new "
        "schema, head set and frozen reward, and "
        "`resolve_frozen_run_shape()` is called here. Partly OUTSIDE the "
        "narrow grant (argument parsing and reporting), partly inside it "
        "(the reward mixing).",
    'train_on_comparison.py::resolve_frozen_run_shape':
        "WP8: kappa and the episode length are filled from the frozen "
        "object instead of from flag defaults. New function, INSIDE the "
        "reward-handling grant.",
    'train_on_comparison.py::run_comparison_worker_episode':
        "WP6/WP7: the worker episode builds the new observation and "
        "action shapes. INSIDE the action/reward grant.",
    'train_on_comparison.py::run_comparison_worker_episode._act':
        "WP6 critic-feature switch (`critic_features(...)` into "
        "`get_value`) and the WP7 head-set store (one `relay` head index "
        "replacing five head entries). INSIDE the action/reward grant. "
        "UNDECLARED until D8.26 -- two of the four missing hunks.",
    'train_on_comparison.py::run_comparison_worker_episode._complete_pending':
        "WP6/WP7: the pending transition is completed with the new "
        "action/observation shapes. INSIDE the action/reward grant.",
    'train_on_comparison.py::run_eval_episode':
        "WP7: the pinned-head set comes from `all_pinnable_heads()` "
        "rather than the deleted module constant. INSIDE the "
        "action-handling grant. UNDECLARED until D8.26 -- the hunk in the "
        "function the previous declaration never named at all.",
}

# Regions whose reason is derived rather than written out, because the file's
# whole diff has one cause.  path -> reason applied to every region in it.
FILE_REASONS = {
    'sixg_sim/worker.py':
        'WP6 critic-feature switch and WP7 head-set store: the worker builds '
        'the vectors the new actor/critic split and head set require.',
    'sixg_sim/main.py':
        'WP8: kappa and the episode length are published constants, so the '
        'flags are held to them instead of carrying their own defaults '
        '(D8.24).',
    'sixg_sim/heuristic_controller.py':
        'XL-DET baseline: re-expressed against the WP7 head set so the '
        'comparator emits the same actions the simulator still applies.',
    'sixg_sim/mappo_trainer.py':
        'WP6/WP7: the shared actor/critic split, the per-head distributions '
        'and the head list the update loops iterate.',
    'sixg_sim/runinfo.py':
        'WP8: reward_id() joins the provenance record, so a pickle names the '
        'reward that produced it.',
    'sixg_sim/phy_mac_state.py':
        'WP6 observation field the actor reads.',
    'sixg_sim/iops/learning_postcard.py':
        'WP7: a deleted head is no longer carried on the postcard.',
    'sixg_sim/iops/migrate_checkpoint.py':
        'WP6/WP7: deleted -- it migrated checkpoints across a schema this '
        'tree no longer has.',
    'sixg_sim/reward_profile.py':
        'WP8: deleted -- an environment-selected coefficient set is the '
        'drift channel this WP exists to close.',
    'plot_severance_maps.py':
        'WP8/D8.24: the figure script mirrored the training episode length '
        'with its own literal; it reads the published constant now.',
    'test_relay_hysteresis.py':
        'Root contract script: re-based on the WP7 head index, and the '
        'relay-margin comparison is exercised rather than short-circuited '
        '(D8.20).',
    'test_mission_config.py':
        'Root contract script: the WP8 guard no longer requires a reward '
        'profile, since the coefficients are frozen.',
    'test_marl_convergence.py':
        'Root contract script: the WP6 schema and the WP7 head set.',
    'test_relay_mask.py':
        'Root contract script: the WP7 relay head replaces the deleted '
        'per-head masks.',
    'test_xldet.py':
        'Root contract script: the XL-DET comparator under the WP7 heads.',
    'test_ensemble_policy.py':
        'Root contract script: the WP6 schema gate.',
    'tests/test_wp1_routing_graph.py':
        'Shared routing test updated for the WP6 schema id.',
    'tests/test_wp2_parity.py':
        'Shared parity test updated for the WP6/WP7 schema and head set.',
    'tests/data/xldet_frozen_9dec943.py':
        'A frozen COPY of the baseline XL-DET controller, added so the '
        'comparator can be checked against the evaluated tree.',
    'tools/constancy_dump.py':
        'WP8 diagnostic: added by this track.',
    'tools/reward_term_dump.py':
        'WP8 diagnostic: added by this track.',
    'tools/shared_file_edits.py':
        'This generator: added by this track.',
}

_HUNK = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')


def _git(*args):
    out = subprocess.run(('git',) + args, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    if out.returncode != 0:
        raise RuntimeError('git %s failed: %s'
                           % (' '.join(args), out.stderr.decode('utf-8',
                                                                'replace')))
    return out.stdout.decode('utf-8', 'replace')


def is_owned(rel):
    return rel in OWNED or any(rel.startswith(p) for p in OWNED_PREFIXES)


def changed_paths():
    """Every path in the diff, split into source and non-source."""
    names = [p.strip() for p in
             _git('diff', '--name-only', BASE, HEAD_REV).splitlines()
             if p.strip()]
    src = sorted(p for p in names if p.endswith('.py'))
    # SELF-EXCLUSION: the generated document is itself a non-source
    # path in the diff.  Listing it would make the generator's output
    # depend on whether its own output had been committed yet, and no
    # fixed point would exist.  Everything else is listed.
    other = sorted(p for p in names
                   if not p.endswith('.py') and p != DOC)
    return src, other


def hunks(rel):
    """New-file (start, length) of every hunk of one path."""
    text = _git('diff', '-U0', BASE, HEAD_REV, '--', rel)
    out = []
    for line in text.splitlines():
        m = _HUNK.match(line)
        if m:
            start = int(m.group(1))
            length = 1 if m.group(2) is None else int(m.group(2))
            out.append((start, length))
    return out


def _source_at_rev(rel):
    """The text of `rel` at HEAD_REV, or None when it does not exist there."""
    try:
        return _git('show', '%s:%s' % (HEAD_REV, rel))
    except RuntimeError:
        return None


def _spans(path):
    """(start, end, qualname) of every def in a file on disk."""
    with open(path, encoding='utf-8') as f:
        return _spans_source(f.read(), path)


def _spans_source(text, name='<source>'):
    """(start, end, qualname) of every def in `text`, outermost first, so the
    innermost enclosing one wins a linear scan."""
    tree = ast.parse(text, filename=name)
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                out.append((child.lineno, child.end_lineno, name))
                walk(child, name + '.')
            elif isinstance(child, ast.ClassDef):
                name = prefix + child.name
                out.append((child.lineno, child.end_lineno, name))
                walk(child, name + '.')
    walk(tree, '')
    out.sort()
    return out


def regions(rel):
    """{region: [(start, length), ...]} for one path, at HEAD_REV."""
    hs = hunks(rel)
    if not hs:
        return {}
    text = _source_at_rev(rel)
    if text is None:
        return {'(file deleted)': hs}
    spans = _spans_source(text, rel)
    out = {}
    for start, length in hs:
        line = max(1, start)
        name = '<module>'
        for s, e, nm in spans:
            if s <= line <= e:
                name = nm
        out.setdefault(name, []).append((start, length))
    return out


def reason_for(rel, region):
    key = '%s::%s' % (rel, region)
    if key in REASONS:
        return REASONS[key]
    if rel in FILE_REASONS:
        return FILE_REASONS[rel]
    return None


def _fmt_ranges(hs):
    return ' '.join('%d,%d' % (s, n) for s, n in sorted(hs))


def build():
    src, other = changed_paths()
    owned = [p for p in src if is_owned(p)]
    shared = [p for p in src if not is_owned(p)]

    per_path = {}
    for rel in src:
        per_path[rel] = regions(rel)

    n_owned = sum(len(h) for p in owned for h in per_path[p].values())
    n_shared = sum(len(h) for p in shared for h in per_path[p].values())

    missing = []
    for rel in shared:
        for region in per_path[rel]:
            if reason_for(rel, region) is None:
                missing.append('%s::%s' % (rel, region))
    if missing:
        raise SystemExit(
            'no reason recorded for %d region(s): %s\n'
            'Add them to REASONS or FILE_REASONS in tools/shared_file_edits.py'
            % (len(missing), missing))
    live = set()
    for rel in shared:
        for region in per_path[rel]:
            live.add('%s::%s' % (rel, region))
    stale = sorted(k for k in REASONS if k not in live)
    if stale:
        raise SystemExit(
            'REASONS names %d region(s) that are no longer in the diff: %s'
            % (len(stale), stale))
    stale_files = sorted(k for k in FILE_REASONS if k not in shared)
    if stale_files:
        raise SystemExit(
            'FILE_REASONS names %d path(s) that are not declared paths '
            'of this diff: %s' % (len(stale_files), stale_files))

    L = []
    w = L.append
    w('# Shared-file edits, branch `trackAgent`')
    w('')
    w('GENERATED FILE -- do not edit by hand.  Regenerate with:')
    w('')
    w('    python3 tools/shared_file_edits.py --write')
    w('')
    w('Every entry below is derived from `git diff -U0 %s %s`: the ranges'
      % (BASE, HEAD_REV))
    w('are the NEW-file ranges git prints, and the region is the innermost')
    w('`def`/`class` enclosing the range in the file at `%s`, the branch'
      % HEAD_REV)
    w('head the merge into master consumed.  This is a FROZEN record of the')
    w('branch: after the merge `HEAD` also carries the work of master, so the')
    w('diff is pinned to that head and not taken against `HEAD`.')
    w('`tests/test_wp8_shared_edits.py` regenerates this file (and the hunk')
    w('table `%s`) and fails if the committed text and the diff' % DATA)
    w('disagree; a tree without git checks this file against that table, so')
    w('an omitted hunk is caught there too (D8.26; trackAgent merge).')
    w('')
    w('SCOPE: source paths (`*.py`).  Documentation paths are listed by name')
    w('at the end without counts -- this file is itself one of them, so a')
    w('total that included it could not be written down.')
    w('')
    w('## Reconciliation')
    w('')
    w('| set | paths | hunks |')
    w('|---|---|---|')
    w('| owned by this track | %d | %d |' % (len(owned), n_owned))
    w('| NOT owned by this track | %d | %d |' % (len(shared), n_shared))
    w('| **all source paths in the diff** | **%d** | **%d** |'
      % (len(src), n_owned + n_shared))
    w('')
    w('`%d + %d = %d` hunks over `%d + %d = %d` source paths.'
      % (n_owned, n_shared, n_owned + n_shared,
         len(owned), len(shared), len(src)))
    w('')
    w('## Paths this track does NOT own')
    w('')
    w('Each one is a merge surface a human has to adjudicate.  Provenance is')
    w('a mark on the entry, never a reason to leave one out.')
    w('')
    for rel in shared:
        regs = per_path[rel]
        total = sum(len(h) for h in regs.values())
        w('### `%s` -- %d hunk%s' % (rel, total, '' if total == 1 else 's'))
        w('')
        w('| region | hunks | new-file ranges | why |')
        w('|---|---|---|---|')
        for region in sorted(regs):
            hs = regs[region]
            w('| `%s` | %d | %s | %s |'
              % (region, len(hs), _fmt_ranges(hs), reason_for(rel, region)))
        w('')
    w('## Paths this track owns')
    w('')
    w('Listed for the arithmetic above; no merge adjudication is expected.')
    w('')
    w('| path | hunks |')
    w('|---|---|')
    for rel in owned:
        w('| `%s` | %d |' % (rel, sum(len(h) for h in per_path[rel].values())))
    w('')
    w('## Forbidden paths')
    w('')
    w('| path | hunks |')
    w('|---|---|')
    for rel in FORBIDDEN:
        w('| `%s` | %d |' % (rel, len(hunks(rel))))
    w('')
    w('## Non-source paths in the diff')
    w('')
    w('By name, without counts (see SCOPE above).')
    w('')
    for rel in other:
        w('* `%s`' % rel)
    w('')
    return '\n'.join(L)


def hunk_table():
    """Every hunk of every source path in the diff, one row each:
    `path<TAB>region<TAB>start,length`, sorted.  Owned paths included, so the
    owned counts can be checked too."""
    src, _other = changed_paths()
    rows = ['# path\tregion\tstart,length -- every hunk of git diff -U0 %s %s '
            'over the source paths; GENERATED by tools/shared_file_edits.py '
            '--write, do not edit' % (BASE, HEAD_REV)]
    for rel in src:
        for region, hs in sorted(regions(rel).items()):
            for start, length in sorted(hs):
                rows.append('%s\t%s\t%d,%d' % (rel, region, start, length))
    return '\n'.join(rows) + '\n'


def _write_lf(rel, text):
    data = text.encode('utf-8')
    assert b'\r' not in data
    with open(os.path.join(ROOT, *rel.split('/')), 'wb') as f:
        f.write(data)


def main(argv):
    if argv and argv[0] == '--check':
        doc = argv[1] if len(argv) > 1 else DOC
        bad = 0
        for rel, want in ((doc, build()), (DATA, hunk_table())):
            with open(os.path.join(ROOT, *rel.split('/')), 'rb') as f:
                on_disk = f.read().decode('utf-8')
            if on_disk != want:
                sys.stderr.write('%s is STALE: regenerate with --write\n' % rel)
                bad += 1
            else:
                sys.stdout.write('%s matches the diff\n' % rel)
        return 1 if bad else 0
    if argv and argv[0] == '--write':
        _write_lf(DOC, build())
        _write_lf(DATA, hunk_table())
        sys.stdout.write('wrote %s and %s\n' % (DOC, DATA))
        return 0
    # stdout in BINARY: a text-mode stdout on Windows turns every LF into
    # CRLF, which is how the committed document once went CRLF
    sys.stdout.buffer.write(build().encode('utf-8'))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
