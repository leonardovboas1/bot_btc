#!/usr/bin/env bash
# One-shot setup on a fresh Ubuntu 22.04/24.04 server (x86_64 or ARM). Run as the default user:
#   curl -fsSL https://raw.githubusercontent.com/leonardovboas1/bot_btc/main/deploy/setup.sh | bash
# Installs the repo in ~/bot_trade and three systemd services that restart on crash and on boot.
set -euo pipefail
APP="$HOME/bot_trade"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3 python3-venv python3-pip sqlite3 >/dev/null
# 1 GB machines: add swap so numba compilation never OOMs
if [ "$(free -m | awk '/Mem:/{print $2}')" -lt 2000 ] && ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
[ -d "$APP/.git" ] && git -C "$APP" pull -q || git clone -q https://github.com/leonardovboas1/bot_btc.git "$APP"
python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q -r "$APP/deploy/requirements-server.txt"
mkdir -p "$APP/data/paper" "$APP/data/raw/hyperliquid" "$APP/data/raw/liquidations"
sudo timedatectl set-timezone UTC || true

unit() {  # name module
  sudo tee "/etc/systemd/system/signallab-$1.service" >/dev/null <<UNIT
[Unit]
Description=SignalLab $1
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP
Environment=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
ExecStart=$APP/.venv/bin/python -m $2
Restart=always
RestartSec=15
Nice=10

[Install]
WantedBy=multi-user.target
UNIT
}
unit paper src.live.paper
unit liq src.data.liq_recorder
unit hl src.data.hl_collector
sudo systemctl daemon-reload
echo "Installed. If you are moving an existing paper ledger, upload data/paper/paper.sqlite and strategies.json now,"
echo "then run:  sudo systemctl enable --now signallab-paper signallab-liq signallab-hl"
