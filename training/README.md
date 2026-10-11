# Training: your checklist

You own: dataset in, trained ACT policy out. The teleop teammate records the data and runs the model on the arm.

## Before the cloud link arrives (on your Mac)
1. **Hugging Face:** create an account and a **write** token at https://huggingface.co/settings/tokens
2. **W&B:** create an account and copy your key from https://wandb.ai/authorize
3. Give your HF username to the teleop teammate. Their dataset must be named `YOUR_HF_USER/so101_screwdriver_bin`, so they log in to `hf` with **your** token on the GMKtec. (Or make an HF org and both of you join it.)

## When AMD gives you the MI300X Jupyter link
4. Open AMD's `training-models-on-rocm.ipynb` and run its setup cells in order: GPU check → ffmpeg 7 → LeRobot v0.4.1 → wandb login → HF login (paste your tokens into those cells yourself).
5. Upload this `training/` folder: use the **Upload Files** button in the left pane (or `git clone` your team repo).
6. Open a terminal (**File → New Launcher → Terminal**). Then:
   ```bash
   cd training
   ```
   ```bash
   bash train.sh smoke
   ```
   This does 1,000 steps on a public SO-101 pick-place dataset. Watch the loss fall in W&B (project `lerobot`). **If this works, your whole pipeline works.**

## When the teammate says the dataset is uploaded
7. Check it (this works on your Mac too):
   ```bash
   python3 check_dataset.py YOUR_HF_USER/so101_screwdriver_bin
   ```
   If it prints FAIL or WARN, tell the teammate what to fix before you train.
8. Train:
   ```bash
   bash train.sh real so101_screwdriver_bin
   ```
   That's 20k steps, with a checkpoint every 5k. The model auto-uploads to `YOUR_HF_USER/act_so101_screwdriver_bin` when it finishes.
9. To let the arm team test early while training keeps going:
   ```bash
   bash push_checkpoint.sh act_so101_screwdriver_bin 005000
   ```
   It prints the exact download command to send them.
10. If training stops:
   ```bash
   bash train.sh resume act_so101_screwdriver_bin
   ```

## Iterate
Watch the arm run the policy. Where it fails (an object position, a missed grasp), ask for about 10 more demos of exactly that case. They add them with `--resume=true`, then you retrain.

## Know it for Q&A
ACT = a **ResNet18 CNN** (ImageNet-pretrained) encodes each camera image. Then a **transformer** with a **CVAE** predicts a **chunk of 100 future joint positions** at once, which gives smooth motion and fewer compounding errors. It's trained by imitation (L1 loss on actions plus a KL term) from about 50 human teleop demos.
