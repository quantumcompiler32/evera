#!/usr/bin/env bash
# Copy these scripts from the Mac to ~/pi05 on the GMKtec:  GMK=aup@<GMKTEC_IP> bash pi05/sync.sh
set -euo pipefail
rsync -av --exclude outputs "$(dirname "$0")/" "${GMK:?set GMK=aup@<GMKTEC_IP> first}:pi05/"
