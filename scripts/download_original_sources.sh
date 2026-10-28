# 파일: download_original_sources.sh
#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(pwd)/unitime_sources"
mkdir -p "$BASE_DIR"

declare -A REPOS=(
  ["ego4d"]="ego4d/ego4d"
  ["anet"]="activitynet/anet"
  ["charades"]="charades/charades"
  ["qvhl"]="qvhl/qvhl"
  ["tacos"]="tacos/tacos"
  ["pretrain"]="zeqianli/UniTime-Pretrain"
)

for name repo in "${!REPOS[@]}"; do
  echo "🔽 Downloading $name → ${REPOS[$name]}"
  hf download "${REPOS[$name]}" --repo-type dataset \
      --local-dir "$BASE_DIR/$name"
done

echo "✅ All sources downloaded under $BASE_DIR"
