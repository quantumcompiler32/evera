# AMD Physical AI / LeRobot setup

This repository uses a local Python 3.12 virtual environment for the AMD Physical AI track work.

## Activate the environment

```bash
source .venv/bin/activate
```

## Verify LeRobot

```bash
python -c "import lerobot, torch; print('LeRobot', getattr(lerobot, '__version__', 'installed')); print('PyTorch', torch.__version__); print('MPS available', torch.backends.mps.is_available())"
lerobot-record --help
lerobot-calibrate --help
```

## Installed workflow support

The environment includes LeRobot's `core_scripts`, `training`, and `feetech` extras for recording, calibration, dataset creation, policy training, and SO-101/SO-100 motor support.

This development machine is an Apple M4 Mac, so local acceleration uses Apple's MPS backend when available. AMD GPU acceleration and physical SO-101 access must be verified on the AMD-provided AI PC/cloud environment and robot hardware.

## Next hardware step

When the SO-101 is connected, follow the current LeRobot SO-101 hardware instructions for USB permissions, port discovery, motor calibration, and camera setup before running any motion command.
