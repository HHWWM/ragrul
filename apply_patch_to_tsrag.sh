#!/usr/bin/env bash
set -e
TARGET_REPO="$1"
if [ -z "$TARGET_REPO" ]; then
  echo "Usage: bash apply_patch_to_tsrag.sh /path/to/original/TS-RAG"
  exit 1
fi
cp -r ./TS-RAG/bearing "$TARGET_REPO/"
cp ./TS-RAG/configs/phm2012_rul.yaml "$TARGET_REPO/configs/"
cp ./TS-RAG/script/preprocess_phm2012_rul.sh "$TARGET_REPO/script/"
cp ./TS-RAG/script/train_phm2012_rul.sh "$TARGET_REPO/script/"
cp ./TS-RAG/script/eval_phm2012_rul.sh "$TARGET_REPO/script/"
cp ./TS-RAG/script/thingsboard_phm2012_rul.sh "$TARGET_REPO/script/"
echo "Patch files copied into: $TARGET_REPO"
