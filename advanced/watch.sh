#!/usr/bin/env bash
# Collect live dataset versions while the continuous GPU session stays active.
set -euo pipefail
cd "$(dirname "$0")/.."
particle_deadline=$((SECONDS + ${PARTICLE_WATCH_SECONDS:-19800}))
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
publish_progress() {
  git add advanced/run_state.json advanced_runs/
  if ! git diff --cached --quiet; then
    git commit -m 'Save parametric Kaggle training progress'
    for attempt in 1 2 3; do
      if git push origin HEAD:main; then return 0; fi
      git pull --rebase origin main
    done
    return 1
  fi
}
trap publish_progress EXIT
if [ "${START_NEW:-false}" = "true" ]; then
  python advanced/controller.py --start
  publish_progress
fi
while [ "$SECONDS" -lt "$particle_deadline" ]; do
  # The public checkpoint must be committed before its immutable URL is used.
  if ! python advanced/controller.py --collect-only; then
    # Transient API/download failures must not kill the live collector.
    publish_progress
    particle_status=$(python -c "import json; print(json.load(open('advanced/run_state.json'))['status'])")
    if [ "$particle_status" = 'running' ] || [ "$particle_status" = 'ready' ]; then
      sleep 30
      continue
    fi
    break
  fi
  publish_progress
  if ! python advanced/controller.py; then
    publish_progress
    sleep 30
    continue
  fi
  publish_progress
  particle_status=$(python -c "import json; print(json.load(open('advanced/run_state.json'))['status'])")
  if [ "$particle_status" != 'running' ] && [ "$particle_status" != 'ready' ]; then
    break
  fi
  sleep 30
done
