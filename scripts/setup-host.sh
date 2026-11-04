#!/usr/bin/env bash
# One-time host setup for Ubuntu 22.04 / 24.04 with an NVIDIA GPU.
#  * NVIDIA driver R580 (LTS branch, open kernel modules) -> CUDA 13 capable
#  * Docker Engine + compose plugin (official repo)
#  * NVIDIA Container Toolkit (+ docker runtime config)
#  * ufw: only SSH, HTTP, HTTPS
# Usage: sudo scripts/setup-host.sh        (reboot afterwards if the driver changed)
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
DRIVER_BRANCH="${DRIVER_BRANCH:-580}"
. /etc/os-release
echo "Host: $PRETTY_NAME"
[[ "$VERSION_ID" == "22.04" ]] && echo "NOTE: Ubuntu 22.04 standard support ends April 2027 — plan the move to 24.04 (do-release-upgrade)."

apt-get update
apt-get install -y ca-certificates curl gnupg ufw ubuntu-drivers-common

# --- NVIDIA driver -----------------------------------------------------------
current=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
if [[ -z "$current" || "${current%%.*}" -lt "$DRIVER_BRANCH" ]]; then
  echo "Installing nvidia-driver-${DRIVER_BRANCH}-server-open (current: ${current:-none})"
  apt-get install -y "nvidia-driver-${DRIVER_BRANCH}-server-open" "nvidia-utils-${DRIVER_BRANCH}-server"
  NEED_REBOOT=1
else
  echo "Driver $current is fine (>= $DRIVER_BRANCH)"
fi

# --- Docker Engine -----------------------------------------------------------
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# --- NVIDIA Container Toolkit ------------------------------------------------
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# --- Firewall ----------------------------------------------------------------
# NB: Docker-published ports bypass ufw; that is why only Caddy publishes ports.
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp
ufw --force enable

# --- dirs --------------------------------------------------------------------
install -d -o 1000 -g 1000 -m 2775 /models/hf

echo
echo "RAM: $(free -g | awk '/Mem:/{print $2}') GB (>= 32 GB recommended: vLLM sleep mode parks the LLM weights in RAM)"
if [[ "${NEED_REBOOT:-0}" == 1 ]]; then
  echo ">>> Driver installed — reboot now, then check: nvidia-smi && docker run --rm --gpus all ubuntu nvidia-smi"
else
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi
fi
