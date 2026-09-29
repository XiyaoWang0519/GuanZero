"""Epoch-stamped nvidia-smi samples (utilization, memory, power) until killed or deadline."""
import json
import subprocess
import sys
import time

output, interval, deadline = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
QUERY = "utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"
with open(output, "a") as stream:
    while time.time() < deadline:
        began = time.time()
        try:
            done = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}",
                                   "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10)
            fields = [f.strip() for f in done.stdout.strip().splitlines()[0].split(",")]
            record = dict(epoch=(began + time.time()) / 2, util=float(fields[0]),
                          mem_util=float(fields[1]), mem_used_mib=float(fields[2]),
                          power_w=float(fields[3]) if fields[3] not in ("[N/A]", "") else None,
                          sm_mhz=float(fields[4]) if fields[4] not in ("[N/A]", "") else None)
        except FileNotFoundError:
            stream.write(json.dumps(dict(epoch=time.time(), error="nvidia-smi unavailable")) + "\n")
            break
        except Exception as error:
            record = dict(epoch=time.time(), error=f"{type(error).__name__}: {error}"[:300])
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        time.sleep(max(0.0, interval - (time.time() - began)))
