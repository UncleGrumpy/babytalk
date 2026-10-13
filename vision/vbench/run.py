"""Run vbench on a board and check it against the PC.

  python3 vision/synth.py --check                                  # the models + expected hashes
  python3 vision/vbench/run.py esp32s3 -p /dev/ttyACM0 --flash --mac e0:72:a1:b4:2c:c8
  python3 vision/vbench/run.py esp32s3 -p /dev/ttyACM0             # again, nothing flashed

--flash writes the app (build it first: vision/vbench/idf.sh <chip> build) and both models, after
checking the board's MAC is --mac (or $BOARD_MAC): several boards may share a port. Without it the
board is only reset. Results: vision/out/vbench-<chip>.jsonl (appended), and a summary next to
ESP-DL's published times for the same networks.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import serial

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
IDF = os.environ.get("IDF_PATH", str(Path.home() / "build/lvgl_micropython/lib/esp-idf"))
MODELS = {"mobilenetv2_224": 0x200000, "mobilefacenet_112": 0x600000}
# ESP-DL's model zoo, milliseconds per image (int8): MobileNetV2 (imagenet_cls), MobileFaceNet (human_face_recognition MFN_S8_V1)
ESPDL = {("esp32s3", "mobilenetv2_224"): 856, ("esp32p4", "mobilenetv2_224"): 226,
         ("esp32s3", "mobilefacenet_112"): 246.7, ("esp32p4", "mobilefacenet_112"): 82.1}


def esptool(chip, port, *args):
    cmd = f". '{IDF}/export.sh' >/dev/null 2>&1; python -m esptool --chip {chip} -p {port} " + " ".join(args)
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chip", choices=["esp32s3", "esp32p4"])
    ap.add_argument("-p", "--port", default="/dev/ttyACM0")
    ap.add_argument("--flash", action="store_true")
    ap.add_argument("--mac", default=os.environ.get("BOARD_MAC"))
    ap.add_argument("--build", help="build directory (default build-<chip>; build-esp32s3-quad for the Pager)")
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()

    expect = {}
    for name in MODELS:
        j = OUT / f"{name}.json"
        if not j.exists():
            sys.exit(f"no {j}: run python3 vision/synth.py --check first")
        expect[name] = json.loads(j.read_text())

    r = esptool(a.chip, a.port, "read_mac")
    mac = next((l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("MAC:")), None)
    if not mac:
        sys.exit(f"no {a.chip} answering on {a.port}:\n{r.stdout[-500:]}{r.stderr[-500:]}")
    print(f"{a.chip} {mac} on {a.port}")
    if a.flash:
        if not a.mac or a.mac.lower() != mac.lower():
            sys.exit(f"not flashing: the board's MAC is {mac}, --mac/$BOARD_MAC is {a.mac}")
        b = HERE / (a.build or f"build-{a.chip}")
        boot = "0x2000" if a.chip == "esp32p4" else "0x0"
        files = [boot, b / "bootloader/bootloader.bin", "0x8000", b / "partition_table/partition-table.bin",
                 "0x10000", b / "vbench.bin"]
        for name, off in MODELS.items():
            files += [hex(off), OUT / f"{name}.mmrt"]
        r = esptool(a.chip, a.port, "-b 921600 write_flash", *map(str, files))
        if r.returncode:
            sys.exit(r.stdout[-2000:] + r.stderr[-2000:])
        print("flashed: app + models")
    else:
        esptool(a.chip, a.port, "--after hard_reset chip_id")

    rows, log = [], OUT / f"vbench-{a.chip}.jsonl"
    deadline = time.time() + a.timeout
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = a.port, 115200, 1
    s.dtr = s.rts = False  # opening the port mustn't pull EN or IO0 (a reset, or the ROM bootloader)
    s.open()
    with s, log.open("a") as f:
        while time.time() < deadline:
            line = s.readline().decode(errors="replace").strip()
            if not line:
                continue
            if line.startswith("@@R "):
                rec = json.loads(line[4:])
                rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                f.write(json.dumps(rec) + "\n")
                rows.append(rec)
                print(line[4:])
            elif line.startswith("@@DONE"):
                break
            elif line.startswith("@@") or "mmrt" in line or "abort" in line.lower() or "panic" in line.lower():
                print(line[:300])
        else:
            print("timed out waiting for @@DONE")

    print("\nmodel               weights cores    ms   ESP-DL  speed-up  output")
    for rec in rows:
        if "ms_min" not in rec:
            continue
        name = rec["model"]
        ok = rec["fnv"] == expect[name]["out_fnv"]
        base = ESPDL.get((a.chip, name))
        print(f"{name:20s}{rec['weights']:8s}{rec['cores']:5d}{rec['ms_min']:7.1f}{base or 0:9.1f}"
              f"{(base / rec['ms_min']) if base else 0:9.1f}x  {'matches the PC' if ok else 'DIFFERS from the PC'}")
    for rec in rows:
        if rec.get("test") == "dwconv2d" and "fails" in rec:
            print(f"2-D depthwise kernel vs reference: {'all bit-exact' if not rec['fails'] else str(rec['fails']) + ' cases differ'}")


if __name__ == "__main__":
    main()
