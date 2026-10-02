#!/usr/bin/env bash
# Greedy-decode an artifact with NInfer, then teacher-force the same tokens through the reference.
#   e2e_decode.sh ARTIFACT CHECKPOINT "prompt" [max_new] [extra ninfer args...]
set -euo pipefail
artifact=$1 checkpoint=$2 prompt=$3 max_new=${4:-16}
shift $(( $# < 4 ? $# : 4 ))
here=$(cd "$(dirname "$0")" && pwd)
ninfer=${NINFER:-$HOME/Projects/ninfer-flash-next/build/apps/ninfer}
template=${TEMPLATE:-$HOME/Projects/ninfer-flash-next/tools/chat_templates/qwen3_8.jinja}
log=$(mktemp)
"$ninfer" "$artifact" --prompt "$prompt" --max-context 2048 --kv-capacity 2048 \
    --max-new "$max_new" --greedy --no-thinking --print-token-ids --kv-dtype bf16 "$@" \
    >"$log.out" 2>"$log"
ids=$(grep -E '^tokens +generated ids' "$log" | sed -E 's/^tokens +generated ids *//')
echo "NInfer ids: $ids"
echo "NInfer text: $(cat "$log.out")"
"$here/.venv/bin/python" "$here/compare_decode.py" --checkpoint "$checkpoint" \
    --template "$template" --prompt "$prompt" --generated "$ids" --out "$here/decode_ref.pt"
