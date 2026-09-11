#!/bin/zsh
set -e
cd "$(dirname "$0")"

echo "========================================"
echo "      TPR TERMINAL V4 ALPHA 4.1 INSTALLER"
echo "========================================"

PYTHON=""
for candidate in python3.14 python3.13 python3.12; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.12-3.14 was not found. Install Python from python.org first."
  read "?Press Return to close..."
  exit 1
fi

echo "Using $($PYTHON --version)"

# Use a new environment name so a broken legacy .venv cannot block installation.
if [ ! -x ".venv-tpr/bin/python" ]; then
  "$PYTHON" -m venv .venv-tpr
fi

source .venv-tpr/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  echo ""
  echo "Opening secure configuration wizard..."
  python configure.py
fi

python doctor.py || true
chmod +x install_mac.command run_mac.command

echo ""
echo "Installation complete. Double-click run_mac.command to launch."
read "?Press Return to close..."
