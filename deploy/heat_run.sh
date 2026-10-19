#!/usr/bin/env bash
# Run the phone at a fixed setting for a while and log its health every 30 s, read from the gateway.
# Stops early if the battery gets hotter than MAX_BATTERY_C or emptier than MIN_BATTERY_PCT.
# Usage: heat_run.sh <label> <seconds> <motion_gate true|false>   (writes measurements/heat-<label>.csv)
# Needs the gateway running on the laptop and adb reverse tcp:8000 tcp:8000. Preview stays off.
set -euo pipefail
label="$1"; duration_s="$2"; motion_gate="$3"
MAX_BATTERY_C="${MAX_BATTERY_C:-45}"
MIN_BATTERY_PCT="${MIN_BATTERY_PCT:-25}"
out="$(cd "$(dirname "$0")/.." && pwd)/measurements/heat-${label}.csv"

adb shell am broadcast -n com.doorcounter.app/.ConfigReceiver --ez preview false --ez motion_gate "${motion_gate}" \
  | tr -d '\r' | tail -1
adb shell input keyevent 223  # screen off, as in use
echo "elapsed_s,fps,mean_inference_ms,cpu_c,battery_c,battery_pct,charging" > "${out}"
start=$(date +%s)
stop_reason="time"
while [ $(( $(date +%s) - start )) -lt "${duration_s}" ]; do
  sleep 30
  row=$(curl -s --max-time 5 http://127.0.0.1:8000/api/latest | python3 -c '
import json, sys
h = json.load(sys.stdin).get("health")
if h:
    print(f"{h[\"fps\"]:.2f},{1000 * h[\"mean_inference_s\"]:.0f},{h[\"cpu_temp_c\"]:.1f},{h[\"battery_temp_c\"]:.1f},{h[\"battery_level_pct\"]},{h[\"is_charging\"]}")
' || true)
  [ -z "${row}" ] && continue
  echo "$(( $(date +%s) - start )),${row}" >> "${out}"
  battery_c=$(echo "${row}" | cut -d, -f4); battery_pct=$(echo "${row}" | cut -d, -f5)
  if python3 -c "import sys; sys.exit(0 if float('${battery_c}') > ${MAX_BATTERY_C} else 1)"; then stop_reason="battery above ${MAX_BATTERY_C} C"; break; fi
  if [ "${battery_pct}" -lt "${MIN_BATTERY_PCT}" ]; then stop_reason="battery below ${MIN_BATTERY_PCT}%"; break; fi
done
adb shell am force-stop com.doorcounter.app
echo "stopped: ${stop_reason}; wrote ${out}"
