"""WP0: pool cap, hash-seed guard, provenance record, per-pass meter."""
import contextlib
import io
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sixg_sim import runinfo  # noqa: E402


def _mkdtemp(prefix):
    """Temporary directory inside the tree (output/tmp_tests): the server
    rules forbid writing outside ~/MARL_run_9, /tmp included."""
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def test_pool_workers_default_and_cap():
    assert runinfo.pool_workers(32, env={}) == 32
    assert runinfo.pool_workers(32, env={'MARL_POOL_WORKERS': ''}) == 32
    assert runinfo.pool_workers(32, env={'MARL_POOL_WORKERS': ' 8 '}) == 8
    assert runinfo.pool_workers(4, env={'MARL_POOL_WORKERS': '8'}) == 4
    assert runinfo.pool_workers(0, env={'MARL_POOL_WORKERS': '8'}) == 0
    assert runinfo.pool_workers(0, env={}) == 0


def test_pool_workers_rejects_bad_values():
    for bad in ('0', '-3', 'eight', '2.5'):
        try:
            runinfo.pool_workers(16, env={'MARL_POOL_WORKERS': bad})
        except ValueError:
            continue
        raise AssertionError('MARL_POOL_WORKERS=%r accepted' % bad)


def test_pool_workers_reads_process_environment():
    old = os.environ.get('MARL_POOL_WORKERS')
    try:
        os.environ['MARL_POOL_WORKERS'] = '3'
        assert runinfo.pool_workers(10) == 3
        del os.environ['MARL_POOL_WORKERS']
        assert runinfo.pool_workers(10) == 10
    finally:
        if old is None:
            os.environ.pop('MARL_POOL_WORKERS', None)
        else:
            os.environ['MARL_POOL_WORKERS'] = old


def test_hashseed_guard():
    buf = io.StringIO()
    assert runinfo.check_hashseed('t', env={'PYTHONHASHSEED': '0'}, stream=buf)
    assert buf.getvalue() == ''
    for env in ({'PYTHONHASHSEED': '1'}, {'PYTHONHASHSEED': 'random'}, {}):
        buf = io.StringIO()
        assert not runinfo.check_hashseed('tag-x', env=env, stream=buf)
        out = buf.getvalue()
        assert 'WARNING [tag-x]' in out and 'PYTHONHASHSEED' in out
        assert out.count('!' * 78) == 2


def _make_tree(files):
    d = _mkdtemp('wp0_tree_')
    for rel, content in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as f:
            f.write(content)
    return d


def test_provenance_from_synced_tree():
    d = _make_tree({'a.py': b'print(1)\n', 'sub/b.txt': b'x'})
    try:
        with open(os.path.join(d, '.synced_head'), 'w') as f:
            f.write('abc1234\n')
        with open(os.path.join(d, '.synced_files'), 'w') as f:
            f.write('a.py\nsub/b.txt\n')
        old = os.environ.get('MARL_WP0_TEST_VAR')
        os.environ['MARL_WP0_TEST_VAR'] = 'yes'
        try:
            rec = runinfo.provenance(extra={'driver': 'test'}, root=d)
        finally:
            if old is None:
                del os.environ['MARL_WP0_TEST_VAR']
            else:
                os.environ['MARL_WP0_TEST_VAR'] = old
        assert rec['tree_head'] == 'abc1234'
        assert rec['tree_head_source'] == '.synced_head'
        assert rec['tree_files'] == 2 and rec['tree_files_missing'] == []
        assert rec['tree_files_excluded'] == []
        assert len(rec['tree_sha256']) == 64
        assert rec['provenance_complete'] is True
        assert 'INCOMPLETE' not in runinfo.provenance_summary(rec)
        assert rec['pythonhashseed'] == os.environ.get('PYTHONHASHSEED')
        assert rec['marl_env'].get('MARL_WP0_TEST_VAR') == 'yes'
        assert all(k.startswith('MARL_') for k in rec['marl_env'])
        assert rec['schema_id']
        assert rec['driver'] == 'test'
        for k in ('argv', 'python', 'host', 'created_utc'):
            assert k in rec
        assert 'head=abc1234' in runinfo.provenance_summary(rec)

        # the digest tracks file content, not just the head
        digest = rec['tree_sha256']
        with open(os.path.join(d, 'a.py'), 'wb') as f:
            f.write(b'print(2)\n')
        runinfo._DIGEST_CACHE.clear()
        assert runinfo.tree_digest(d)['tree_sha256'] != digest
        os.remove(os.path.join(d, 'sub', 'b.txt'))
        runinfo._DIGEST_CACHE.clear()
        assert runinfo.tree_digest(d)['tree_files_missing'] == ['sub/b.txt']
    finally:
        runinfo._DIGEST_CACHE.clear()
        shutil.rmtree(d, ignore_errors=True)


def test_digest_ignores_adapted_checkpoint_artefacts():
    ck = 'ce_ck_s1234/policy_final.pt'
    art = 'ce_ck_s1234/policy_final_adapted_recovery_freeze_seed58.pt'
    art2 = 'ce_ck_s1234/policy_final_adapted_recovery_seed82.pt'
    d = _make_tree({'a.py': b'1', ck: b'weights', art: b'adapted-v1'})
    try:
        with open(os.path.join(d, '.synced_files'), 'w') as f:
            f.write('\n'.join(['a.py', ck, art, art2]) + '\n')
        dig = runinfo.tree_digest(d)
        assert dig['tree_files'] == 2, dig
        assert dig['tree_files_excluded'] == [art, art2]
        assert dig['tree_files_missing'] == []      # art2 absent: not missing
        base = dig['tree_sha256']
        # a run rewrites the artefact (and creates another): digest unchanged
        with open(os.path.join(d, art), 'wb') as f:
            f.write(b'adapted-v2')
        with open(os.path.join(d, art2), 'wb') as f:
            f.write(b'new')
        runinfo._DIGEST_CACHE.clear()
        assert runinfo.tree_digest(d)['tree_sha256'] == base
        # the checkpoint itself still counts
        with open(os.path.join(d, ck), 'wb') as f:
            f.write(b'weights-changed')
        runinfo._DIGEST_CACHE.clear()
        assert runinfo.tree_digest(d)['tree_sha256'] != base
        assert runinfo._digest_excluded('x\\y_adapted_recovery_seed1.pt')
        assert not runinfo._digest_excluded('adapted/policy_final.pt')
    finally:
        runinfo._DIGEST_CACHE.clear()
        shutil.rmtree(d, ignore_errors=True)


def test_provenance_without_sync_or_git():
    d = _make_tree({'a.py': b'1'})
    try:
        head, source = runinfo.tree_head(d)
        dig = runinfo.tree_digest(d)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rec = runinfo.provenance(root=d)
            runinfo.provenance(root=d)            # warned once per tree
        if runinfo._git(d, 'rev-parse', '--show-toplevel'):
            # temp dir inside a git work tree (laptop checkout): git fallback
            assert source == 'git' and head
            assert dig['tree_files_source'] == 'git ls-files'
            assert dig['tree_files'] == 0          # output/ is untracked
            assert rec['provenance_complete'] is True
            assert buf.getvalue() == ''
        else:
            # server copy: ~/MARL_run_9 is not a git work tree
            assert head is None and source == 'unavailable'
            assert dig['tree_sha256'] is None
            assert rec['provenance_complete'] is False
            out = buf.getvalue()
            assert out.count('WARNING [provenance]') == 1, out
            assert 'tree_head and tree_sha256 unavailable' in out
            assert 'INCOMPLETE' in runinfo.provenance_summary(rec)
        # the helper itself, independent of the environment
        s = io.StringIO()
        bogus = {'tree_root': d + '/bogus', 'tree_head': 'abc',
                 'tree_sha256': None}
        assert runinfo.warn_incomplete(bogus, stream=s) is True
        assert 'tree_sha256 unavailable' in s.getvalue()
        assert runinfo.warn_incomplete({'tree_head': 'a', 'tree_sha256': 'b'},
                                       stream=s) is False
    finally:
        runinfo._DIGEST_CACHE.clear()
        shutil.rmtree(d, ignore_errors=True)


def test_schema_id_placeholder_until_wp6():
    from sixg_sim import agent
    expected = getattr(agent, 'SCHEMA_ID', runinfo.SCHEMA_ID_PLACEHOLDER)
    assert runinfo.schema_id() == expected


def test_pass_meter():
    tok = runinfo.pass_meter_start()
    blob = bytearray(64 * 1024 * 1024)      # touch ~64 MB
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    st = runinfo.pass_meter_stop(tok)
    del blob
    assert st['wall_s'] >= 0 and st['cpu_s'] >= 0
    assert st['pid'] == os.getpid()
    if sys.platform.startswith('linux'):
        assert tok['hwm_reset'], 'clear_refs reset failed'
        assert st['peak_rss_source'].startswith('VmHWM')
        assert st['peak_rss_mb'] >= 64, st
        assert st['rss_end_mb'] is not None and st['pss_end_mb'] is not None
        # the reset really lowers the mark: a second, small pass must not
        # report the 64 MB spike of the first
        tok2 = runinfo.pass_meter_start()
        st2 = runinfo.pass_meter_stop(tok2)
        assert st2['peak_rss_mb'] < st['peak_rss_mb'], (st, st2)


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
