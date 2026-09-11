#!/bin/zsh
cd "$(dirname "$0")"

if [ ! -x ".venv-tpr/bin/python" ]; then
  echo "TPR environment is missing. Run install_mac.command first."
  read "?Press Return to close..."
  exit 1
fi

source .venv-tpr/bin/activate
python doctor.py || {
  echo "Doctor found a required configuration problem."
  read "?Press Return to close..."
  exit 1
}

echo "Starting TPR Terminal V4 Alpha 4.1..."
python main.py
STATUS=$?
echo ""
echo "Bot stopped with status $STATUS."
read "?Press Return to close..."
