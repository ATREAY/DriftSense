#!/bin/bash
# Picks the highest-priority node with at least one currently-idle GPU:
# A100 > V100 > P100, falling through to the next tier only when the
# current one has zero free GPUs right now (checked live via `scontrol`,
# not just partition state -- a node can show ALLOCATED while still
# having idle GPUs if another job only grabbed some of them).
#
# Usage: node=$(training/select_gpu.sh [--tiers A100,V100]); echo "$node"
# --tiers restricts which GPU types are considered (default: all three).
# Prints one node name to stdout. Falls back to the top-priority
# considered tier (so the job queues there) if nothing is free right now.
set -euo pipefail

TIERS="A100,V100,P100"
if [[ "${1:-}" == "--tiers" ]]; then
  TIERS="$2"
fi

# priority-ordered "TYPE NODE" candidates for this cluster
ALL_CANDIDATES=(
  "A100 dgx-a100-02"
  "V100 dgx-v100-01"
  "P100 dgx-p100-01"
  "P100 cse-node009"
  "P100 cse-node010"
  "P100 cse-node011"
  "P100 cse-node012"
)

CANDIDATES=()
for entry in "${ALL_CANDIDATES[@]}"; do
  type="${entry%% *}"
  if [[ ",${TIERS}," == *",${type},"* ]]; then
    CANDIDATES+=("$entry")
  fi
done
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "[select_gpu] --tiers '$TIERS' matched no candidates" >&2
  exit 1
fi

idle_gpus() {
  local node="$1" info state cfg alloc
  info=$(scontrol show node "$node" 2>/dev/null) || { echo -1; return; }
  state=$(sed -n 's/.*State=\([A-Za-z+*]*\).*/\1/p' <<<"$info" | head -1)
  if [[ "$state" == *DOWN* || "$state" == *DRAIN* || "$state" == *NOT_RESPONDING* ]]; then
    echo -1; return
  fi
  cfg=$(sed -n 's/.*CfgTRES=.*gres\/gpu=\([0-9]*\).*/\1/p' <<<"$info" | head -1)
  alloc=$(sed -n 's/.*AllocTRES=.*gres\/gpu=\([0-9]*\).*/\1/p' <<<"$info" | head -1)
  cfg="${cfg:-0}"; alloc="${alloc:-0}"
  echo $(( cfg - alloc ))
}

for entry in "${CANDIDATES[@]}"; do
  type="${entry%% *}"; node="${entry#* }"
  free=$(idle_gpus "$node")
  echo "[select_gpu] $node ($type): idle=$free" >&2
  if [[ "$free" -gt 0 ]]; then
    echo "$node"
    exit 0
  fi
done

echo "[select_gpu] nothing free right now within tiers [$TIERS]; falling back to top-priority considered node (will queue)" >&2
echo "${CANDIDATES[0]#* }"
