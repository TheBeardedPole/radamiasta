#!/usr/bin/env bash
# One-time setup: a venv for the transcription helpers and the voice models (~33 MB).
# The whisper model (~1.6 GB) downloads itself on the first transcription.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
mkdir -p models
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
[ -f models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx ] ||
  curl -sL "$R/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" | tar xj -C models
[ -f models/wespeaker_en_voxceleb_resnet34_LM.onnx ] ||
  curl -sLo models/wespeaker_en_voxceleb_resnet34_LM.onnx \
    "$R/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
cp -n config.env.example config.env
echo "Ready. Fill in config.env to enable uploads, then: ./radamiasta.py run"
