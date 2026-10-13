# Historical handoff: October 11, 2026 (early morning)

> This is an operator snapshot, not a live status report. Paths, model availability,
> dataset counts, and machine state require rechecking before use.

Read this first if you're picking up on a new computer or in a new Claude session. Setup from scratch is in
[SETUP.md](SETUP.md), and how the pieces connect is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Where things run

| Machine | Address | What's on it |
|---|---|---|
| **NucBox** (GMKtec EVO-X2) | `ssh aup@<GMKTEC_IP>` | Arms, cameras, dashboard, recordings in `~/hackathon/`, π₀.₅ in `~/pi05/`, vision in `~/evera-vision/` |
| AMD Radeon Cloud (Jupyter) | radeon-global.anruicloud.com | Training only. **The disk is wiped on every launch.** Stop the instance when idle (1 credit/h). |
| Any laptop or the iPad | browser | Dashboard at `http://<GMKTEC_IP>:8080`, evera at `/evera/` |

Hugging Face on the NucBox is logged in as **Neevdharod**. Don't replace that login: `record.sh` uploads to `Neevdharod/*`.

## Everyday commands (on the NucBox)

```bash
bash ~/hackathon/robot/start_limelight.sh                 # camera bridges (auto-restart on stalls)
bash ~/hackathon/robot/record.sh status|check|review|upload|practice
bash ~/hackathon/robot/record.sh policy <ckpt_dir> [secs] # run a checkpoint on the follower
```

Restart the dashboard with this one-liner:

```bash
ssh aup@<GMKTEC_IP> 'cd ~/hackathon && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && kill $(cat ~/dashboard.pid); setsid nohup python robot/dashboard.py > ~/dashboard.log 2>&1 < /dev/null & echo $! > ~/dashboard.pid'
```

The dashboard shows live top and wrist cameras, plus START SESSION, DONE/NEXT (Space), REDO (R), STOP (Q),
PRACTICE (teleop without recording), RUN POLICY, Check motors, Mark bad and Upload.

## Datasets

All are 30 fps with `scene` + `wrist` cameras at 320x240.

| Dataset | Hub (`Neevdharod/…`) | NucBox `~/hackathon/data/` | Notes |
|---|---|---|---|
| `so101_pick_place` | 30 eps, private | 101 eps | **Same name, different contents.** `record.sh upload` pushes only the good episodes, so the Hub copy is the clean 30-episode set. The local copy has every episode, with 71 listed in `so101_pick_place.bad`. |
| `so101_pick_place_clean` | — | 30 eps | Local copy of the same 30 good episodes as the Hub's `so101_pick_place` |
| `so101_pick_place_50_80` | 19 eps, private | 19 eps | The good episodes 50–80. All current ACT models and π₀.₅ runs trained on this. |
| `so101_pick_place_90_100` | 11 eps, public | 11 eps | The latest good episodes |
| `so101_pick_place_rejects` | 71 eps, private | 71 eps | The bad episodes, kept for reference |
| `so101_pick_place_snap32` | — | 32 eps | A snapshot made mid-recording for a π₀.₅ test |
| `test_run` | 2 eps, private (plus a public `test_run_20261010_175539`) | 2 eps | Pipeline test |

Things to know:
- **`so101_screwdriver_bin` doesn't exist yet.** It's the default dataset name in `record.sh`, the dashboard and
  `pi05/`, meant for a clean re-record with a fixed bin. Until it exists, pass `DATASET=` explicitly. To train
  on all 30 good episodes, use the Hub's `Neevdharod/so101_pick_place`.
- The object in every recording is a screwdriver, but the `so101_pick_place*` task text says "pick up the
  lego brick and place it in the bin". ACT ignores task text. π₀.₅ doesn't: give a π₀.₅ rollout exactly the
  task text its dataset used.

## Trained models

| Model | Where | Trained on | Notes |
|---|---|---|---|
| `trial_act` | NucBox `~/hackathon/outputs/trial_act` | eps 15–29 | First ACT run, superseded |
| `bc_act_50_80` | NucBox outputs, Hub `Neevdharod/act_so101_pick_place_ep50_80` | eps 50–80 | Plain fine-tune. **Best offline numbers.** |
| `rl_act` | NucBox outputs, Hub `Neevdharod/act_so101_pick_place_awr` | eps 50–80, reward-weighted | Dashboard default and evera's *Hand over* checkpoint |
| π₀.₅ smoke test | NucBox `~/pi05/outputs/pi05_test_run_1010_1833` | `test_run`, 20 steps | Pipeline check only |
| π₀.₅ fine-tunes | Radeon Cloud `/workspace/pi05_outputs` | `so101_pick_place_50_80` | Lost on relaunch unless pushed to the Hub |

The offline comparison is in [training/README.md](../training/README.md#offline-rl-fine-tune-runs-on-the-gmktec-gpu).
The plain fine-tune beat AWR on the held-out episodes, but the dashboard and evera still default to `rl_act`.
Pick one after testing both on the real arm.

## Next steps

1. Run `bc_act_50_80` and `rl_act` on the real arm from the dashboard and record their success rates.
2. Point the dashboard default and `ACT_PICK` in `evera/actions.js` at whichever one wins.
3. Retrain ACT on all 30 good episodes (`Neevdharod/so101_pick_place` on the Hub), not just the 19 in eps 50–80.
4. Push any π₀.₅ checkpoint worth keeping from Radeon Cloud to the Hub before the instance stops.
5. For the hackathon submission: publish code, datasets, models and a demo video (see [HACKATHON.md](HACKATHON.md)).

## Hardware facts (learned the hard way)

- **Follower** (gripper arm) is USB serial `5C4C125836`, **leader** (handle arm) is `5C4C128274`.
- **The gripper overload-locks if it is still squeezing when an episode ends.** That takes motor #6 offline.
  Always open the gripper before DONE. To recover, unplug the follower's power barrel for 5 s. It happened
  on Saturday evening; recording resumed afterwards.
- The whole USB hub has dropped out a few times. Plug the arms straight into the NucBox, and don't touch
  cables while recording.
- The wrist cam runs at 20 fps unless asked for 30. `limelight_zmq.py` now sets 30 fps, and its log prints
  the capture mode on start. Episodes recorded before that fix have a 20 fps wrist feed in a 30 fps dataset.
- **Don't restart the Limelight's vision server.** It once came back on an AprilTag pipeline (grayscale, LED
  on). If that happens, rerun `start_limelight.sh`.
- The NucBox is on Wi-Fi only (~5 MB/s from the Hub). Long SSH sessions sometimes stall, so run long jobs
  with `nohup` and read their logs.

## Recording protocol (what makes the data good)

1. **One taped-down bin** in the same spot every episode. A consistent drop spot matters most, because ACT
   averages demos that disagree.
2. **4–5 taped start spots** for the object, ~10 demos each, with it pointing the same way.
3. The same rest pose at the start and end. **Open the gripper wide** and fully release before DONE.
4. One smooth reach → grab → lift → drop → return. REDO anything fumbled.
5. A clear table, no people or hands in view, the same lighting. Camera profiles are locked by `start_limelight.sh`.
6. Every ~10 episodes, run `bash ~/hackathon/robot/record.sh review`. It flags NO_GRASP / NOT_RELEASED /
   PLACE_OFF / START_NOT_REST and writes contact sheets to `~/hackathon/review/`. Aim for a drop-spot pan
   spread under 5°.
