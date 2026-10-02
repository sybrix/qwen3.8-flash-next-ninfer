# Serving scripts

These scripts serve Qwen3.8-Flash-Next with NInfer on port 8000. They also let one GPU switch
between it and a second NInfer model. Here that second model is Qwen3.8-27B, run from a separate
upstream NInfer checkout. The two can't run together: Flash-Next needs about 80 GiB of the 96 GiB
card.

| File | Purpose |
|---|---|
| `swap-model.sh` | `./swap-model.sh flash-next \| 27b \| status`. Stops whatever NInfer server owns :8000, starts the other one, and waits until it is healthy. If Flash-Next fails to start, it shows the log tail and restores the 27B. |
| `start-flash-next.sh` | Starts the Flash-Next server: 262K context, FP8 KV, 2 concurrent requests, MTP speculative decoding with 3 draft tokens. Run it in the foreground or with `--background`. The comments explain each flag and the memory budget. |
| `start-ninfer.sh` | Starts the Qwen3.8-27B NVFP4 server (vision, DFlash2 speculation, 262K context). `--check` sends a real image to prove vision is live. |
| `stop-ninfer.sh` | Stops any `ninfer-serve` cleanly: SIGTERM, wait, escalate to SIGKILL, then confirm the port and VRAM are free. It identifies the process by its real executable, not by matching the command line. |
| `ninfer.service` | systemd unit that runs `start-ninfer.sh` at boot. Replace `YOUR_USER` and the paths before installing it to `/etc/systemd/system/`. |

## Configuration

The paths are environment variables with defaults:

| Variable | Default | Meaning |
|---|---|---|
| `FLASH_NEXT_DIR` | `$HOME/Projects/ninfer-flash-next` | Build of `ninfer-flash-next/`; holds `models/qwen3_8_flash_next_nvfp4_mtp.ninfer`. |
| `NINFER_DIR` | `$HOME/Projects/ninfer` | Upstream NInfer checkout serving the 27B model. |

`swap-model.sh` uses `sudo systemctl` to stop and start `ninfer.service`. A reboot always brings
back the 27B: Flash-Next runs as a background process, not as a systemd unit.

`ninfer-serve` answers requests whatever `model` name they carry. Use `./swap-model.sh status` to
see which model is actually serving.
