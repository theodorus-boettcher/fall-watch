#!/usr/bin/env bash
# Set up a factory-reset S5 (no accounts on it) as a dedicated door counter:
# install the app, make it device owner, grant it the camera, hide the apps in hide-list.txt.
# Safe to run again. Usage: provision.sh [path/to/app.apk]
# Undo the hiding:  provision.sh --unhide      Give up device owner:  provision.sh --release
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
pkg=com.doorcounter.app

send() {
  adb shell am broadcast -n "${pkg}/.ProvisionReceiver" "$@" < /dev/null | tr -d '\r' | sed -n 's/.*data="\(.*\)"/\1/p'
}
hide_list() {
  grep -vE '^\s*(#|$)' "${here}/hide-list.txt" | paste -sd, -
}

case "${1:-}" in
  --unhide) send --es op unhide --esa packages "$(hide_list)"; exit 0 ;;
  --release) send --es op release; exit 0 ;;
esac

apk="${1:-${here}/../android/app/build/outputs/apk/debug/app-debug.apk}"
adb install -r "${apk}" < /dev/null
if ! adb shell dumpsys device_policy < /dev/null | grep -q "${pkg}"; then
  adb shell dpm set-device-owner "${pkg}/.AdminReceiver" < /dev/null
fi
send --es op grant-camera
send --es op hide --esa packages "$(hide_list)"
