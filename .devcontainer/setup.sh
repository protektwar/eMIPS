#!/usr/bin/env bash
# The eMIPS dev container (a GitHub Codespace): what the guide uses.
#  - Python and a virtual environment with eMIPS and PyQt6 (emips, emips-gui)
#  - the Qt libraries the window needs on the container's desktop
#  - Ubuntu 24.04's MIPS GCC 12 (the guide's numbers come from it), its
#    RISC-V GCC, and gdb-multiarch for the GDB labs
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-venv python3-pip \
    gcc-12-mips-linux-gnu binutils-mips-linux-gnu \
    gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu \
    gdb-multiarch \
    libgl1 libegl1 libfontconfig1 libdbus-1-3 libxkbcommon-x11-0 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
    fonts-dejavu-core fonts-noto-core
# the guide's C labs use mips-linux-gnu-gcc: GCC 12, as the guide quotes it
if ! command -v mips-linux-gnu-gcc >/dev/null; then
    sudo ln -s mips-linux-gnu-gcc-12 /usr/bin/mips-linux-gnu-gcc
fi
sudo apt-get clean

python3 -m venv "$HOME/.emips-venv"
"$HOME/.emips-venv/bin/pip" install --upgrade pip
"$HOME/.emips-venv/bin/pip" install -e ".[gui]"

if ! grep -q emips-venv "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<'EOF'

# eMIPS: python3 and emips / emips-gui from its virtual environment
export PATH="$HOME/.emips-venv/bin:$PATH"
if [[ $- == *i* ]]; then
    echo "eMIPS: 'emips-gui' opens the window on the desktop (the Ports tab:"
    echo "       open port 6080 in the browser); 'python3 -m emips --help' for the"
    echo "       command line; the user guide is docs/guide/index.html."
fi
EOF
fi
