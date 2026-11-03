# vbench: MMRT's 2-D ops on a board

A speed test for MMRT on images: two networks with the shapes of real ones and random weights
(`../synth.py`), timed on an ESP32-S3 or ESP32-P4 and checked against the PC bit for bit.

| model | input | multiply-adds | weights | ESP-DL's time, S3 / P4 |
|---|---|---|---|---|
| `mobilenetv2_224`: MobileNetV2 1.0, an image classifier | 224 x 224 x 3 | 316 M | 3.5 MB | 856 / 226 ms |
| `mobilefacenet_112`: MobileFaceNet, a face to a 128-d identity vector | 112 x 112 x 3 | 222 M | 1.0 MB | 247 / 82 ms |

(ESP-DL's figures are from its model zoo: `imagenet_cls` MobileNetV2 and `human_face_recognition`
MFN_S8_V1. Our MobileNetV2 has 5% more multiply-adds: channel counts padded to multiples of 16.)

```sh
python3 vision/synth.py --check                  # the models, checked on the PC -> vision/out/
vision/vbench/idf.sh esp32s3 build               # or esp32p4
python3 vision/vbench/run.py esp32s3 -p /dev/ttyACM0 --flash --mac <the board's MAC>
```

`run.py` flashes the app and both models (after checking the MAC), resets the board, and
prints each configuration: weights read from flash, weights cached in PSRAM, and on the S3 one
core vs two. Every output's hash must match the PC's. Results are appended to
`vision/out/vbench-<chip>.jsonl`.

The S3 build is for 8 MB octal PSRAM (the Waveshare S3-CAM). For a quad-PSRAM board (the
LilyGO T-LoRa Pager): `idf.sh esp32s3-quad build`, then `run.py esp32s3 --build build-esp32s3-quad ...`.

First results (T-LoRa Pager, quad PSRAM, 2026-10-13): bit-exact, but 0.6-1.1x ESP-DL's times.
The activations live in PSRAM and the run is bandwidth-bound (the residual adds alone take
78 / 118 ms; two cores are slower than one). Next: run each inverted-residual block in row bands
so the expanded tensor stays in internal RAM.

What's new in MMRT for this (format version 3):

- `MMRT_DWCONV2D`: a K x K depthwise conv on an image stored as `[H*W][C]`. Reference
  (`mmrt_ref.c`), portable (`mmrt_port.c`), and vector kernels for the S3 (`mmrt_s3_dw2d_px`)
  and the P4 (`mmrt_p4_dw2d_px`, which checks itself against the portable one first).
- `MMRT_IM2COL`: a first layer's K x K patches as rows, so that the existing 1x1 kernels run
  it as the full K x K convolution.
- `mmrt_op_t.W`: the input's width, for the 2-D ops. Everything else is unchanged: the 1x1
  convs, the global average pool (`MEAN`) and the residual adds were already there.
