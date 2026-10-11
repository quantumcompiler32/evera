#!/usr/bin/env bash
# Copy these scripts from the Mac to ~/pi05 on the GMKtec:  bash pi05/sync.sh
set -euo pipefail
rsync -av --exclude outputs "$(dirname "$0")/" "${GMK:-aup@169.233.72.19}:pi05/"
