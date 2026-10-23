#!/usr/bin/env bash
# Set up a factory-reset S5 (no accounts on it) as a dedicated fall watch camera:
# install the app, make it device owner, grant it the camera, hide the apps in hide-list.txt, start it.
# Safe to run again. Usage: provision.sh [path/to/app.apk]   (default: the signed release build)
# Undo the hiding:  provision.sh --unhide      Give up device owner:  provision.sh --release
# Wi-Fi: a router can have two radios (2.4 and 5 GHz) under one name, and a weak one makes frames, and so alerts,
# run late. At the final spot, compare them:  adb shell dumpsys wifi | grep -A12 'Latest scan results'
# and pin the stronger one (bssid any undoes it):  adb shell am broadcast -n com.doorcounter.app/.ProvisionReceiver \
#   --es op pin-wifi --es ssid <wifi name> --es bssid <address>
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
  --release) send --es op release; sleep 5; exit 0 ;;  # the admin role goes in the background; uninstall fails before
esac

apk="${1:-${here}/../android/app/build/outputs/apk/release/app-release.apk}"
adb install -r "${apk}" < /dev/null
if ! adb shell dumpsys device_policy < /dev/null | grep -q "${pkg}"; then
  adb shell dpm set-device-owner "${pkg}/.AdminReceiver" < /dev/null
fi
send --es op grant-camera
send --es op hide --esa packages "$(hide_list)"
adb shell am start -n "${pkg}/.MainActivity" < /dev/null > /dev/null  # installing stopped the running app
echo "started"
