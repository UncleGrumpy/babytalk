#!/usr/bin/env bash
# idf.py for one chip, each in its own build directory:
#   ./idf.sh esp32s3 build
#   ./idf.sh esp32p4 -p /dev/ttyACM0 flash
#   ./idf.sh esp32s3-quad build      (quad PSRAM: the T-LoRa Pager)
# IDF_PATH defaults to the v5.5.1 tree the other firmwares use.
set -e
TARGET="$1"; shift
BUILD="build-$TARGET"
DEFAULTS="sdkconfig.defaults"
case "$TARGET" in
  esp32s3|esp32p4) ;;
  esp32s3-quad) TARGET=esp32s3; DEFAULTS="sdkconfig.defaults;sdkconfig.quad" ;;
  *) echo "usage: $0 esp32s3|esp32s3-quad|esp32p4 <idf.py args>" >&2; exit 2 ;;
esac
export IDF_PATH="${IDF_PATH:-$HOME/build/lvgl_micropython/lib/esp-idf}"
. "$IDF_PATH/export.sh" >/dev/null 2>&1
cd "$(dirname "$0")"
exec idf.py -B "$BUILD" -D IDF_TARGET="$TARGET" -D SDKCONFIG="$BUILD/sdkconfig" -D SDKCONFIG_DEFAULTS="$DEFAULTS" "$@"
