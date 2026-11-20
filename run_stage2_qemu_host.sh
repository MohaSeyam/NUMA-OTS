#!/usr/bin/env bash
# run_stage2_qemu_host.sh (Stage 2 — Host launcher for 4-Node Virtual NUMA in QEMU)
#
# Sets up an Ubuntu 24.04 cloud image WITH a cloud-init seed ISO (so you can log in
# with user: ubuntu / pass: ubuntu), configures 4 virtual NUMA nodes (8 vCPUs, 4 GiB RAM)
# AND an explicit ACPI SLIT distance table (10 local, 18 1-hop, 22 2-hop), and shares
# this project folder with the guest via VirtFS (-virtfs).
#
# Usage on Ubuntu / WSL2:
#   chmod +x run_stage2_qemu_host.sh run_stage2_inside_guest.sh
#   ./run_stage2_qemu_host.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VM_DIR="${HOME}/qemu_numa"
mkdir -p "$VM_DIR"
cd "$VM_DIR"

IMG="ubuntu-24.04-cloud.img"
SEED="seed.img"

if ! command -v qemu-system-x86_64 >/dev/null 2>&1 || ! command -v cloud-localds >/dev/null 2>&1; then
    echo "Installing QEMU and cloud-image-utils..."
    sudo apt-get update
    sudo apt-get install -y qemu-system-x86 qemu-utils cloud-image-utils wget
fi

if [ ! -f "$IMG" ]; then
    echo "Downloading Ubuntu 24.04 cloud image..."
    wget -O "$IMG" https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img
    qemu-img resize "$IMG" 10G
fi

if [ ! -f "$SEED" ]; then
    echo "Creating cloud-init seed ISO (login: ubuntu / password: ubuntu)..."
    cat > user-data <<'EOF'
#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: users, admin
    home: /home/ubuntu
    shell: /bin/bash
    lock_passwd: false
chpasswd:
  list: |
    ubuntu:ubuntu
  expire: false
ssh_pwauth: true
runcmd:
  - mkdir -p /mnt/numa_ots
  - mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/numa_ots || true
EOF
    cat > meta-data <<'EOF'
instance-id: numa-ots-qemu-4node
local-hostname: numa-ots-vm
EOF
    cloud-localds "$SEED" user-data meta-data
fi

KVM_FLAG=""
if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
    KVM_FLAG="-enable-kvm -cpu host"
    echo "KVM acceleration detected and enabled."
else
    echo "Running without KVM (pure emulation; boot takes ~1-2 minutes)."
fi

echo "========================================================================"
echo " Booting 4-Node Virtual NUMA Guest (login: ubuntu / password: ubuntu)"
echo " Once logged in, run:"
echo "   sudo mount -t 9p -o trans=virtio hostshare /mnt && cd /mnt && ./run_stage2_inside_guest.sh"
echo " To exit QEMU: press Ctrl+A then X"
echo "========================================================================"

# shellcheck disable=SC2086
qemu-system-x86_64 $KVM_FLAG \
    -m 4G -smp 8,sockets=4,cores=2,threads=1 \
    -object memory-backend-ram,size=1G,id=m0 \
    -object memory-backend-ram,size=1G,id=m1 \
    -object memory-backend-ram,size=1G,id=m2 \
    -object memory-backend-ram,size=1G,id=m3 \
    -numa node,nodeid=0,memdev=m0,cpus=0-1 \
    -numa node,nodeid=1,memdev=m1,cpus=2-3 \
    -numa node,nodeid=2,memdev=m2,cpus=4-5 \
    -numa node,nodeid=3,memdev=m3,cpus=6-7 \
    -numa dist,src=0,dst=1,val=18 -numa dist,src=0,dst=2,val=18 -numa dist,src=0,dst=3,val=22 \
    -numa dist,src=1,dst=2,val=18 -numa dist,src=1,dst=3,val=18 -numa dist,src=2,dst=3,val=18 \
    -drive file="$IMG",format=qcow2 \
    -drive file="$SEED",format=raw \
    -virtfs local,path="$PROJECT_DIR",mount_tag=hostshare,security_model=none,id=hostshare \
    -net nic -net user \
    -nographic
