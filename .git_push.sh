#!/bin/bash
set -e
cd /home/bro/WanMotion
git add -A
git -c user.name="huaweianime001-spec" -c user.email="huaweianime001-spec@users.noreply.github.com" \
  commit -m "Import Wan2.2 with CLIP motion transfer (WanMotion).

Based on Wan2.2 TI2V-5B: motion encoder, adapter, training and inference
scripts, docs, and example assets (cat/corgi/kling driver video)."
git push -u origin main
