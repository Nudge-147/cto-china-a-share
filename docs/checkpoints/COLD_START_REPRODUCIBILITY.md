# Cold-start reproducibility checkpoint

Verification date: 2026-08-09 EDT

## Isolation procedure

The candidate repository was exported while explicitly excluding `.git/`, `data/`, virtual
environments, caches, ZIP bundles, `kaggle.json`, PDFs, machine launch agents, and other ignored
artifacts. The export was initialized and committed as a temporary Git repository, then cloned into
a second temporary directory. All commands below ran from that clone, not from the working tree.

- Clone-test path: `/tmp/a-share-cold-start.X0oENu/clone` (ephemeral audit path)
- Candidate tracked files: 155
- Largest non-Git file: 35,442,465 bytes, below GitHub's 100 MB hard limit
- Python runtime: the already isolated project `.venv-5min` built from the documented requirements

## Command

```bash
bash scripts/cold_start_smoke.sh /absolute/path/to/.venv-5min/bin/python
```

The script verified the principal dependency imports, compiled `src/`, `tests/`, and root Python
entry points, opened eleven CLI `--help` paths, and ran the full test suite.

## Result

- Source compilation: PASS
- CLI smoke checks: PASS
- Unit tests: **41/41 PASS**
- Final marker: `COLD_START_SMOKE=PASS`

The test intentionally does not download remote vendor data. Full data regeneration is rate-limited
and separately documented in the README; final Stage-4 figures use the verified local V17 primary
copy under `data/archive/`.
