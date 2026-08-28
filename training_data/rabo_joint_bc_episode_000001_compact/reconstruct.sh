#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out="rabo_joint_bc_episode_000001_compact.tar.gz"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
: > "$tmp"
for i in $(seq -w 0 23); do
  head -c 18000 "parts/part-0${i}.b64" >> "$tmp"
done
head -c 12540 "parts/part-024.b64" >> "$tmp"
base64 -d "$tmp" > "$out"
echo "12fb8805af192f4cffbaa6f1ee6bbc8d9ae1276367c0e7babffbbde2c69dd938  $out" | sha256sum -c -
echo "reconstructed: $(pwd)/$out"
