#!/usr/bin/env bash
#
# install_slurm.sh
# -----------------
# Idempotent helper for provisioning a single-node Slurm controller+compute setup.
# Intended for quick experimentation on a standalone host. Run it with root
# privileges (e.g. `sudo ./install_slurm.sh`) after the base environment has been
# prepared by scripts/setup.sh.

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "❌ This script must be run as root (try: sudo $0)" >&2
  exit 1
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_user() {
  local name="$1"
  if ! id -u "${name}" >/dev/null 2>&1; then
    log "Creating user '${name}'"
    useradd --system --create-home --shell /usr/sbin/nologin "${name}"
  fi
}

require_systemd() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "❌ systemctl is unavailable; this script requires a systemd-based host." >&2
    exit 1
  fi
  local init_comm
  init_comm="$(ps -p 1 -o comm= 2>/dev/null || true)"
  if [[ "${init_comm}" != "systemd" ]]; then
    echo "❌ PID 1 is '${init_comm:-unknown}', not systemd. Run on a systemd host or container configured with systemd." >&2
    exit 1
  fi
}

install_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing Slurm dependencies via apt-get"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    local packages=(
      slurm-wlm
      slurmctld
      slurmd
      munge
      libmunge2
    )
    if apt-cache show munge-tools >/dev/null 2>&1; then
      packages+=(munge-tools)
    else
      log "munge-tools package not found; skipping (munge-keygen not required)."
    fi
    apt-get install -y "${packages[@]}"
  elif command -v dnf >/dev/null 2>&1; then
    log "Installing Slurm dependencies via dnf"
    dnf install -y \
      slurm \
      slurm-slurmd \
      slurm-slurmctld \
      munge \
      munge-libs \
      munge-devel
  else
    echo "❌ Unsupported distribution: need apt-get or dnf" >&2
    exit 1
  fi
}

setup_munge() {
  log "Configuring Munge authentication"
  install -d -m 0700 -o munge -g munge /etc/munge
  if [[ ! -f /etc/munge/munge.key ]]; then
    log "Generating /etc/munge/munge.key"
    dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key status=none
    chown munge:munge /etc/munge/munge.key
    chmod 0400 /etc/munge/munge.key
  fi
  systemctl enable --now munge
}

write_slurm_conf() {
  local conf_dir="/etc/slurm"
  local slurm_conf="${conf_dir}/slurm.conf"
  local hostname; hostname="$(hostname)"
  local cpu_count; cpu_count="$(nproc)"

  log "Writing ${slurm_conf}"
  install -d -o root -g root -m 0755 "${conf_dir}"
  cat >"${slurm_conf}" <<EOF
ClusterName=hsdt-lightning
SlurmctldHost=${hostname}
SlurmctldPort=6817
SlurmdPort=6818
AuthType=auth/munge
SlurmUser=slurm
SlurmdUser=slurm
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SwitchType=switch/none
MpiDefault=none
ProctrackType=proctrack/cgroup
ReturnToService=1
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/none

NodeName=${hostname} CPUs=${cpu_count} State=UNKNOWN
PartitionName=main Nodes=${hostname} Default=YES MaxTime=INFINITE State=UP
EOF
  chown slurm:slurm "${slurm_conf}"
  chmod 0644 "${slurm_conf}"
}

prepare_state_dirs() {
  log "Preparing Slurm spool directories"
  install -d -m 0755 -o slurm -g slurm /var/spool/slurmctld
  install -d -m 0755 -o slurm -g slurm /var/spool/slurmd
  install -d -m 0755 -o slurm -g slurm /var/log/slurm
}

start_slurm() {
  log "Enabling and starting Slurm services"
  systemctl enable --now slurmctld
  systemctl enable --now slurmd
}

post_install_summary() {
  cat <<'EOF'

✔ Slurm installation complete.

Quick checks:
  sudo systemctl status munge
  sudo systemctl status slurmctld
  sudo systemctl status slurmd
  sinfo          # cluster view
  srun hostname  # submit a simple job

To customize the cluster:
  - Edit /etc/slurm/slurm.conf (add nodes/partitions, adjust resources).
  - Regenerate /etc/munge/munge.key and restart munge/slurm if you build a multi-node cluster.

EOF
  echo "Next steps:"
  echo "  sinfo          # cluster view"
  echo "  srun hostname  # submit a simple job"
  echo ""
}

main() {
  install_packages
  ensure_user slurm
  require_systemd
  setup_munge
  write_slurm_conf
  prepare_state_dirs
  start_slurm
  post_install_summary
}

main "$@"
