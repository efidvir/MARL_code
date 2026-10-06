#!/usr/bin/env python3
"""Digest of a saved policy checkpoint (RERUN_SPEC WP-DET.3).

    tools/policy_digest.py CKPT [CKPT ...]
    tools/policy_digest.py --weights-only CKPT      # just the weights sha256

WHY THIS EXISTS.  WP-DET.3 asks for "an identical sha256 of policy_final.pt"
across worker counts.  The file's sha256 cannot be identical, and must not be
made identical: WP0 requires every checkpoint to carry a provenance record,
and that record contains the process id, the creation timestamp and the
argv (which necessarily differs, because each gate run writes to its own
directory).  Those three fields change the FILE without changing the POLICY.

So the digest that is compared is over the WEIGHTS: every tensor of the
state dict, in sorted key order, with its name, dtype and shape, then its
raw bytes.  Two checkpoints with the same `weights_sha256` hold the same
policy bit for bit.  It is computed over EVERY tensor the checkpoint's
state dict holds -- a checkpoint that carries a second tensor dict the
digest would silently ignore, or no tensors at all, is refused rather than
digested into a constant (an empty digest would be the same for every run
and the gate would pass vacuously).

`meta_sha256` covers the metadata that must be reproducible.  The fields
that differ between two gate runs BY CONSTRUCTION are removed first, and
every one of them is printed so a reader sees exactly what was allowed to
differ:

  * provenance.pid, provenance.created_utc     -- the process and the clock
  * provenance.argv, and the three path entries of provenance.args
    (save_policy_path, checkpoint_dir, output_dir) -- each gate run writes
    to its OWN directory, so these carry that directory's name
  * provenance.marl_env['MARL_POOL_WORKERS']   -- the gate's own variable:
    comparing runs at different worker counts is the whole point

Everything else stays in, so a differing `meta_sha256` is a REAL difference
(a different episode count, reward, schema id, tree sha256 or any other
MARL_* variable) and not the noise it was before D-DET.11.

Output, one block per checkpoint:

    <path>
      weights_sha256 <64 hex>   tensors=<n> params=<n>
      meta_sha256    <64 hex>
      volatile       pid=<n> created_utc=<t> workers=<n> argv_tail=<...>
"""
import argparse
import hashlib
import json
import os
import sys

# provenance keys dropped from meta_sha256 (see the module docstring)
VOLATILE = ('pid', 'created_utc', 'argv')
# provenance['args'] keys that hold this run's own output paths
VOLATILE_ARGS = ('save_policy_path', 'checkpoint_dir', 'output_dir')
# provenance['marl_env'] keys the gate varies on purpose
VOLATILE_ENV = ('MARL_POOL_WORKERS',)
TENSOR_DICT_KEYS = ('state_dict', 'model_state_dict')


def _is_tensor_dict(v):
    return (isinstance(v, dict) and len(v) > 0
            and all(hasattr(t, 'detach') for t in v.values()))


def _state_dict(ckpt):
    if isinstance(ckpt, dict):
        found = [k for k in TENSOR_DICT_KEYS if _is_tensor_dict(ckpt.get(k))]
        # Any OTHER tensor dict in the checkpoint would be left out of the
        # digest without a word, so two checkpoints differing only in it
        # would compare equal.  Refuse instead of digesting part of the file.
        extra = [k for k, v in ckpt.items()
                 if k not in TENSOR_DICT_KEYS and _is_tensor_dict(v)]
        if found:
            # A second RECOGNISED tensor dict is exactly as invisible as an
            # unrecognised one: only found[0] would ever be hashed, so two
            # checkpoints differing only in the other one compare EQUAL.
            # `extra` alone did not cover that case -- a checkpoint holding
            # BOTH state_dict and model_state_dict was silently digested on
            # its first half while this tool's docstring said such a file is
            # refused (measured during WP-DET-AUDIT-REVIEW).  Refuse every
            # tensor dict the digest would drop, recognised or not.
            ignored = sorted(extra) + list(found[1:])
            if ignored:
                raise SystemExit(
                    'checkpoint carries tensor dicts the weights digest '
                    'would ignore: %s (digesting %s only would make the '
                    'comparison partial)' % (', '.join(ignored), found[0]))
            return ckpt[found[0]]
        if _is_tensor_dict(ckpt):        # a bare state dict
            return ckpt
    raise SystemExit('not a recognised checkpoint: no state_dict')


def weights_sha256(ckpt):
    """sha256 over (name, dtype, shape, raw bytes) of every tensor, in
    sorted key order.  Returns (hexdigest, n_tensors, n_params).

    The field framing is deliberately left exactly as it was when the
    D-DET.10 gate digests were recorded, so those digests still reproduce;
    the names and shapes are fixed by the actor's architecture, so no two
    state dicts of this policy can frame to the same byte string.
    """
    sd = _state_dict(ckpt)
    h = hashlib.sha256()
    n_par = 0
    for name in sorted(sd):
        t = sd[name].detach().cpu().contiguous()
        h.update(name.encode('utf-8'))
        h.update(str(t.dtype).encode('utf-8'))
        h.update(str(tuple(t.shape)).encode('utf-8'))
        h.update(t.numpy().tobytes())
        n_par += int(t.numel())
    if not sd or n_par == 0:
        # sha256 of nothing is a CONSTANT: every run would "agree".
        raise SystemExit('refusing to digest a checkpoint with no weights '
                         '(%d tensors, %d parameters): the digest would be '
                         'the same for every run' % (len(sd), n_par))
    return h.hexdigest(), len(sd), n_par


def _reproducible_meta(ckpt):
    """The metadata with the by-construction-volatile fields removed."""
    meta = dict((ckpt.get('metadata') or {})) if isinstance(ckpt, dict) else {}
    prov = dict(meta.get('provenance') or {})
    for k in VOLATILE:
        prov.pop(k, None)
    if isinstance(prov.get('args'), dict):
        prov['args'] = {k: v for k, v in prov['args'].items()
                        if k not in VOLATILE_ARGS}
    if isinstance(prov.get('marl_env'), dict):
        prov['marl_env'] = {k: v for k, v in prov['marl_env'].items()
                            if k not in VOLATILE_ENV}
    meta['provenance'] = prov
    return meta


def meta_sha256(ckpt):
    """sha256 of the metadata that is expected to be reproducible."""
    blob = json.dumps(_reproducible_meta(ckpt), sort_keys=True,
                      default=str).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def volatile_fields(ckpt):
    prov = ((ckpt.get('metadata') or {}).get('provenance') or {}) \
        if isinstance(ckpt, dict) else {}
    argv = prov.get('argv') or []
    env = prov.get('marl_env') or {}
    return {'pid': prov.get('pid'),
            'created_utc': prov.get('created_utc'),
            'workers': env.get('MARL_POOL_WORKERS'),
            'argv_tail': ' '.join(str(a) for a in argv[-2:])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--weights-only', action='store_true',
                    help='print only the weights sha256, one per line')
    args = ap.parse_args(argv)

    import torch
    rc = 0
    for path in args.checkpoints:
        if not os.path.exists(path):
            print('%s\n  MISSING' % path)
            rc = 1
            continue
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        w, n_t, n_p = weights_sha256(ckpt)
        if args.weights_only:
            print(w)
            continue
        print(path)
        print('  weights_sha256 %s   tensors=%d params=%d' % (w, n_t, n_p))
        print('  meta_sha256    %s' % meta_sha256(ckpt))
        v = volatile_fields(ckpt)
        print('  volatile       pid=%s created_utc=%s workers=%s argv_tail=%s'
              % (v['pid'], v['created_utc'], v['workers'], v['argv_tail']))
    return rc


if __name__ == '__main__':
    sys.exit(main())
