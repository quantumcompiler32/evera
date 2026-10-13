# Contributing

Use a focused branch and explain the problem, changed behavior, and evidence in each
pull request. Preserve deployment paths and existing research unless a migration is
explicitly part of the change. Issues are tracked in `quantumcompiler32/evera`.

Run `python3 scripts/check_repository.py` with Python 3.12+ and Node.js 22+ before
submitting. These checks need no robot, API credentials, GPU or Python ML packages.
For inference changes, also run `vision/test_segmenter.py` with external weights and
five saved frames; that validates output shape, not recognition accuracy.

Separate implemented code, offline checks, and attended hardware validation in your
report. Never describe UI animation or training loss as physical task success.
Preserve append-only class IDs and camera contracts. Keep complete recording sessions
in one split, and identify unreviewed labels in every evaluation report.

Do not commit API tokens, browser storage exports, raw recordings, model weights,
virtual environments, or generated presentations. The robot service can move hardware;
run it only in a supervised bench environment. Consult the operator before calibration
or physical rollout. Existing calibration backups are bench-specific.

Dependencies are currently documented per workflow, not locked across machines. Avoid
installing or upgrading GPU packages as part of a documentation or UI change.
