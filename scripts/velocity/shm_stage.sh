#!/usr/bin/env bash
# Compatibility name for kits written against the older bench-kit/shm_stage.sh.
# All staging, caches, guards and sync live in gz_job_env.sh; there is no NFS
# fallback. Kits keep calling _gz_shm_start_sync_loop/_gz_shm_stop_sync_loop.
GZ_STAGE_RUN=1
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gz_job_env.sh"
_gz_shm_start_sync_loop() { gz_start_sync; }
_gz_shm_stop_sync_loop() { gz_final_sync; }
_gz_shm_sync_back() { gz_sync_back; }
