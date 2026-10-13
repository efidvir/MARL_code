# Redactions

None. The rules of the published R2 release (host names; absolute paths that carry a user or
a machine name) match nothing in `docs/PREREGISTRATION_R3.md`.

## What this branch is

Three files, published before any pass of the R3-lite confirmatory run: `docs/PREREGISTRATION_R3.md`, the
pre-registration as frozen at commit `c6c5de5` of the run-chain repository;
`records/R3/FREEZE_R3_c6c5de5.sha256`, the SHA-256 of every one of the 208 tracked files of that commit,
written on the compute host at freeze 1; and this list. No result is here: nothing had been run
on the block when the branch was pushed. The documents and the code the pre-registration cites
are not published here; the digest file holds the sha256 of each, so that each can be checked
against it when it is released with the results.

## How to check

The sha256 of `docs/PREREGISTRATION_R3.md` as stored in this branch is

    72eeef6d20ae60f23d3de15d0b488fc7ddfde5fa8ed844150a3672a66c396b09

(take the raw file, or `git cat-file -p <commit>:docs/PREREGISTRATION_R3.md`; a checkout that converts line
endings changes the bytes). It is the value the digest file lists for that path: the published file is the frozen file.
