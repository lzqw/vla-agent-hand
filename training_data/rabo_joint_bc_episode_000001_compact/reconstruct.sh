#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out="rabo_joint_bc_episode_000001_compact.tar.gz"
cat parts/part-*.b64 | tr -d '\n\r' | base64 -d > "$out"
echo "12fb8805af192f4cffbaa6f1ee6bbc8d9ae1276367c0e7babffbbde2c69dd938  $out" | sha256sum -c -
echo "reconstructed: $(pwd)/$out"
