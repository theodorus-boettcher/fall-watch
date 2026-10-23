#!/usr/bin/env bash
# Set up any Android phone (6.0 or newer, able to run 32-bit apps) as a fall watch camera, over adb:
# install the app, grant it the camera, point it at the laptop gateway and start it. Preview stays off.
# Usage: setup_phone.sh <gateway url, e.g. http://<laptop-ip>:8000> [path/to/app.apk]
# For a phone that does only this job, provision.sh also makes the app device owner and hides other apps.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
pkg=com.fallwatch.app
[ $# -ge 1 ] || { echo "usage: setup_phone.sh <gateway url, e.g. http://<laptop-ip>:8000> [path/to/app.apk]" >&2; exit 2; }
gateway="$1"
apk="${2:-${here}/../android/app/build/outputs/apk/release/app-release.apk}"

# Android 15 and newer refuse apps that target Android 6, as this one does on purpose, unless told otherwise.
bypass=""
if [ "$(adb shell getprop ro.build.version.sdk < /dev/null | tr -d '\r')" -ge 35 ]; then
  bypass="--bypass-low-target-sdk-block"
fi
adb install -r ${bypass} "${apk}" < /dev/null
adb shell pm grant "${pkg}" android.permission.CAMERA < /dev/null
# The settings receiver saves and (re)starts the app; rotation, threads and model can be added the same way.
adb shell am broadcast -n "${pkg}/.ConfigReceiver" --es gateway "${gateway}" --ez preview false < /dev/null \
  | tr -d '\r' | tail -1
