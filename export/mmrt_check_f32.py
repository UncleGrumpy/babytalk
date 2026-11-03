"""Check mmrt's float32 ops (mmrt/mmrt_f32.c) against NumPy in float64, on random data with
awkward shapes (row counts that aren't multiples of 4, one row, one head).

  uv run mmrt_check_f32.py
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "mmrt" / "build" / "libmmrt_f32.so"
PF = ctypes.POINTER(ctypes.c_float)


def build():
    LIB.parent.mkdir(exist_ok=True)
    subprocess.check_call(["gcc", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC", "-o", str(LIB),
                           str(ROOT / "mmrt" / "mmrt_f32.c"), "-lm"])
    return ctypes.CDLL(str(LIB))


def f32(a):
    return np.ascontiguousarray(a, np.float32)


def ptr(a):
    return a.ctypes.data_as(PF)


def main():
    lib = build()
    rng = np.random.default_rng(0)
    worst = {}

    def check(name, got, want):
        err = float(np.abs(got - want).max() / max(1.0, np.abs(want).max()))
        worst[name] = max(worst.get(name, 0.0), err)

    for T, N, C in [(1, 16, 8), (3, 5, 7), (4, 64, 81), (21, 192, 64), (21, 64, 256), (13, 1, 64)]:
        W, b, x = f32(rng.normal(size=(N, C))), f32(rng.normal(size=N)), f32(rng.normal(size=(T, C)))
        y = np.zeros((T, N), np.float32)
        lib.mmrt_f32_linear(ptr(W), ptr(b), ptr(x), T, N, C, ptr(y))
        check("linear", y, x.astype(np.float64) @ W.T.astype(np.float64) + b)
        lib.mmrt_f32_linear(ptr(W), None, ptr(x), T, N, C, ptr(y))
        check("linear (no bias)", y, x.astype(np.float64) @ W.T.astype(np.float64))

    for T, C in [(1, 64), (21, 64), (4, 128)]:
        x, w, b = f32(rng.normal(3, 2, size=(T, C))), f32(rng.normal(size=C)), f32(rng.normal(size=C))
        y = np.zeros_like(x)
        lib.mmrt_f32_layernorm(ptr(x), ptr(w), ptr(b), T, C, ctypes.c_float(1e-5), ptr(y))
        xd = x.astype(np.float64)
        check("layernorm", y, (xd - xd.mean(1, keepdims=True)) / np.sqrt(xd.var(1, keepdims=True) + 1e-5) * w + b)
        lib.mmrt_f32_layernorm(ptr(x), ptr(w), ptr(b), T, C, ctypes.c_float(1e-5), ptr(x))   # in place
        check("layernorm (in place)", x, y)

    for n in (1, 9, 64):
        x = f32(rng.normal(0, 5, size=n))
        want = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
        lib.mmrt_f32_softmax(ptr(x), n)
        check("softmax", x, want)

    x = f32(rng.normal(size=50))
    want = np.maximum(x, 0)
    lib.mmrt_f32_relu(ptr(x), 50)
    check("relu", x, want)

    for T, D, H in [(1, 64, 4), (21, 64, 4), (9, 32, 1), (5, 48, 3)]:
        qkv = f32(rng.normal(size=(T, 3 * D)))
        out = np.zeros((T, D), np.float32)
        lib.mmrt_f32_attention(ptr(qkv), T, D, H, ptr(out))
        q, k, v = (qkv[:, i * D:(i + 1) * D].astype(np.float64) for i in range(3))
        hd, want = D // H, np.zeros((T, D))
        for h in range(H):
            s = slice(h * hd, (h + 1) * hd)
            a = q[:, s] @ k[:, s].T / np.sqrt(hd)
            a = np.exp(a - a.max(1, keepdims=True))
            want[:, s] = (a / a.sum(1, keepdims=True)) @ v[:, s]
        check("attention", out, want)

    ok = True
    for name, err in worst.items():
        good = err < 1e-5
        ok &= good
        print(f"{name:22s} worst relative error {err:.1e}  {'ok' if good else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
