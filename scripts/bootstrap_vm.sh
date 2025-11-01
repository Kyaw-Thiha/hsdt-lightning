#!/usr/bin/env bash
#
# bootstrap_vm.sh
# ----------------
# Base provisioning for Ubuntu VMs (e.g., Vast.ai full VMs). Installs common
# CLI tooling and development dependencies that complement scripts/setup.sh,
# which handles the Python environment. Run this once on a fresh VM before
# running setup.sh.

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "❌ This script must be run as root (try: sudo $0)" >&2
  exit 1
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_apt_prereqs() {
  log "Refreshing apt package lists"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
}

install_base_packages() {
  log "Installing baseline development packages"
  local packages=(
    build-essential
    git
    curl
    wget
    rsync
    zip
    unzip
    zstd
    tmux
    htop
    jq
    pkg-config
    ca-certificates
    gnupg
    lsb-release
    ripgrep
    tree
  )
  apt-get install -y "${packages[@]}"
}

install_compiler_libs() {
  log "Installing compiler support libraries"
  local packages=(
    libssl-dev
    libffi-dev
    libbz2-dev
    liblzma-dev
    libreadline-dev
    libsqlite3-dev
    libncurses5-dev
    libncursesw5-dev
    libgdbm-dev
    libgdbm-compat-dev
  )
  apt-get install -y "${packages[@]}"
}

enable_docker_if_present() {
  if command -v docker >/dev/null 2>&1 && command -v systemctl >/dev/null 2>&1; then
    log "Ensuring Docker service is enabled"
    systemctl enable --now docker || log "Docker service could not be enabled (non-critical)"
  fi
}

post_install_summary() {
  cat <<'EOF'

✔ Base VM bootstrap complete.

Next steps:
  - Run scripts/setup.sh <SSH_PORT> to install Python 3.11 and project dependencies.
  - Use scripts/install_slurm.sh on hosts configured with systemd if you need Slurm.

EOF
}

main() {
  ensure_apt_prereqs
  install_base_packages
  install_compiler_libs
  enable_docker_if_present
  post_install_summary
}

main "$@"
