#!/usr/bin/env bash
# Reboot the phone, let it settle, then take 3 RAM readings one minute apart.
# Usage: [STOP_APP=1] measure_ram.sh <label>   (writes measurements/ram-<label>.txt)
# STOP_APP=1 stops our reader app right after boot, to measure the phone without it.
set -euo pipefail
label="$1"
out="$(cd "$(dirname "$0")/.." && pwd)/measurements/ram-${label}.txt"

adb reboot
adb wait-for-device
until [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = "1" ]; do sleep 5; done
if [ "${STOP_APP:-0}" = "1" ]; then
  adb shell am force-stop com.doorcounter.app
fi
sleep 300
adb shell input keyevent 223  # screen off, same state for every measurement

{
  echo "label: ${label}"
  echo "phone date: $(adb shell date | tr -d '\r')"
  echo "enabled packages: $(adb shell pm list packages -e | wc -l)"
  for i in 1 2 3; do
    echo "--- sample ${i}, uptime $(adb shell cat /proc/uptime | tr -d '\r') s"
    adb shell grep -E '"MemTotal|MemFree|Buffers|^Cached|SwapTotal|SwapFree"' /proc/meminfo | tr -d '\r'
    adb shell dumpsys meminfo | grep -E "Total RAM|Free RAM|Used RAM|Lost RAM" | tr -d '\r'
    [ "$i" -lt 3 ] && sleep 60
  done
} > "${out}"
echo "wrote ${out}"
