# Rabo joint BC episode 000001 — compact GitHub package

This is a GitHub-friendly derivative of the validated `episode_000001` BC dataset.

- 850 frames at 5 Hz
- original Parquet joint/proprioception table preserved byte-for-byte
- original metadata preserved
- all three camera streams preserved for all frames, re-encoded at 64x36 to keep the repository payload small
- intended for the fixed-scene BC / joint-VLA smoke test on the 4080 server

The reconstructed archive is:

`rabo_joint_bc_episode_000001_compact.tar.gz`

Expected SHA256:

`12fb8805af192f4cffbaa6f1ee6bbc8d9ae1276367c0e7babffbbde2c69dd938`

Run `bash reconstruct.sh` in this directory after `git pull` to rebuild and verify the archive.

The original full-resolution archive is not committed here; only the camera video resolution is reduced. The state/action data used for joint targets is unchanged.
