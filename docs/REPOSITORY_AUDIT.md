# Repository audit and cleanup

Review date: October 12, 2026. Local starting commit: `f5da8dc`.
Branch: `chore/project-documentation-cleanup`. The GitHub open-issue query returned no open issues.
This is a repository maintenance review, not a complete security audit or hardware certification.

## Scope and findings

The inventory covered all 104 tracked files, component documentation, Python import
usage, shell entry points, browser modules, vendored assets, and local generated
folders. Targeted behavioral review focused on the browser backend, robot dispatch,
and segmentation dataset/training workflow. No exact duplicate tracked files were
found. No code was deleted merely because it looked experimental or lacked callers:
most Python files are standalone CLI entry points.

| Finding | Action |
|---|---|
| Root overview implied broad working tool handovers and live arm telemetry | Replaced with implementation/status table and explicit simulation and checkpoint boundaries |
| Missing fast, hardware-free entry path | Added clone/serve simulator instructions and a source-based Mermaid diagram |
| Vision notes mixed historical three-class proposals with current six-class target | Clarified chronology; retained original data/research evidence and class IDs |
| Generated presentations cluttered Git status | Ignored local build/output folders; preserved their contents on disk |
| Runtime and training roles were hard to navigate | Added docs index, capture-tool guide, contribution guide and dependency inventory below |
| Capture and selection success claimed a trained model was ready | Added job-specific completion messages |
| Validation/test instances satisfied the new-object training prerequisite | Count new-object training instances separately; regression test rejects evaluation-only data |
| Failed status requests retained timeout callbacks | Clear polling timeout in `finally`; test preserves real-connection/offline behavior |
| No repeatable offline validation command or CI | Added syntax/regression runner and GitHub Actions workflow |
| Licensing was ambiguous | Added attribution inventory; left original-code licensing to contributors |

## Dependencies and organization

Keep existing top-level directories: deployment scripts depend on them. Renaming to
`src/` would add migration risk without improving module boundaries. Existing
Mask R-CNN, wire-data import, and SAM scripts are distinct research paths, not proven
unused duplicates. None of their dependencies were removed.

| Workflow | Direct dependencies / environment |
|---|---|
| Browser simulator | Python static server; vendored three.js; no package installation |
| Offline checks | Python standard library, Bash, Node.js |
| Robot dashboard / camera bridges | NumPy, OpenCV, pyzmq; LeRobot/Feetech and Linux hardware support for operations |
| ACT / π₀.₅ | LeRobot, target-compatible PyTorch/ROCm, dataset/video dependencies; matplotlib for plots; pyarrow for snapshots |
| CPU ONNX decoder | NumPy, OpenCV, ONNX Runtime |
| Capture tools | NumPy, OpenCV, pyzmq |
| Segmentation training / proposals | Ultralytics or Torchvision/PyTorch, Pillow, SAM; datasets/pycocotools for WireSeg import |

There is no consolidated dependency manifest or reproducible cross-machine lockfile.
The existing setup scripts use different historical LeRobot/ROCm environments.
Pinning a new global dependency set without testing the AMD target would be misleading;
retain the documented bench setup and validate a target-specific lock in follow-up.

## Verification in this cleanup

- `python3 scripts/check_repository.py`: passed syntax checks for 39 Python files,
  12 shell scripts and 9 JavaScript files, plus four Python and one Node regression tests.
- `.venv/bin/python -m pip check`: no broken requirements in the existing local
  environment. This does not verify the NucBox or GPU compatibility.
- `git diff --check`: passed.
- Local Markdown target check: passed after adding this audit.
- Inspected the existing interface screenshot; it visibly identifies Simulator mode.
- Local simulator HTTP request returned 200.
- Browser smoke-test attempt timed out connecting to the local preview; interactive
  rendering and touch behavior were not verified in this run.
- Limited credential-pattern scan of tracked text found no matching private-key,
  GitHub-token, Hugging Face-token or API-key literals. Git history and runtime
  browser storage were not audited; this is not a guarantee that secrets never existed.

CI is configured but has not run on GitHub. No model weights, saved test frames,
connected arms, target GPU, or cloud voice credentials were used. ONNX output checks,
held-out recognition accuracy, attended physical rollout and voice integration remain
unverified here. Existing training numbers remain historical, not new measurements.

## Remaining engineering priorities

1. Establish reproducible, separate robot/training/CPU environments on the AMD target.
2. Review complete session labels and report held-out class metrics. Guided mode accepts
   unreviewed proposals; old validation labels remain provisional. Source sessions and
   selected copies must not be assigned to different splits.
3. Train and validate a checkpoint per supported physical action. Empty UI checkpoint
   fields currently select the dashboard default, not an independently learned skill.
4. Add authenticated deployment before exposing the robot dashboard beyond a trusted
   bench network. Browser keys live in local storage; avoid shared demo profiles.
5. Harden robot process lifecycle and failure reporting with hardware-isolated tests.
   For example, `record.sh` currently exits zero after policy rollout even if the runner
   fails; PTY cleanup and concurrent one-shot jobs also merit focused review.
6. Reconcile extensible dashboard class names with the ONNX decoder's explicit supported
   class orders before deploying new classes. Do not silently change the class contract.
7. Verify complete third-party license notices and choose an original-code license with
   all contributors before presenting the repository as freely reusable open source.
8. Add a consented hardware demo recording and per-contributor responsibilities when
   evidence is available. Do not substitute simulator footage for hardware success.

No files or Git history were deleted and no hardware commands were run. Changes
are delivered on the separate `chore/project-documentation-cleanup` branch.
