"""Vision-shaped .mmrt models with random weights, for timing MMRT's 2-D ops on a board and
checking them on the PC. Not trained: the shapes and the arithmetic are real, the weights are
noise, scaled layer by layer so the activations look like a real int8 network's (not all
zero, not all saturated) and every sum stays inside the vector units' 20-bit accumulators.

  python3 vision/synth.py                 -> vision/out/{mobilenetv2_224,mobilefacenet_112}.mmrt
  python3 vision/synth.py --check         + run each through the C executor (mmrt/mmrt.c) on
                                            the PC, reference and portable kernels, and compare
                                            every tensor with this script's NumPy forward pass

Each model gets a .json beside it: shapes, multiply-adds per op, and the FNV-1a hash of the
output for the board's input (vbench_input()), which the board's `run` prints for comparison.
"""
import argparse
import ctypes
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"

MAGIC, VERSION = 0x54524D4D, 3
DWCONV, CONV1X1, MEAN, LUT, MUL, ADD, DWCONV2D, IM2COL = range(1, 9)
NONE16, NONE32 = 0xFFFF, 0xFFFFFFFF
HDR = struct.Struct("<16I")
TENSOR = struct.Struct("<Hbx")
OP = struct.Struct("<BBBBbbbbbB2xHHHHII")  # mmrt_op_t: ..., wfmt, pad, in0, in1, out, W, w_off, b_off
assert HDR.size == 64 and TENSOR.size == 4 and OP.size == 28
ACC_LIMIT = 1 << 19  # the S3's and P4's QACC lanes are 20-bit


def r16(n):
    return (n + 15) // 16 * 16


def lcg_bytes(n, seed=12345):
    """The board's input (vbench.c: vbench_input): x[i] = top byte of s = s * 1664525 + 1013904223."""
    out = np.empty(n, np.int8)
    s = seed
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = np.int8(np.uint8(s >> 24))
    return out


def fnv1a(b):
    h = 2166136261
    for x in bytes(b):
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def out8(acc, shift, relu):
    v = (acc + (1 << (shift - 1))) >> shift if shift > 0 else acc << -shift
    if relu:
        v = np.maximum(v, 0)
    return np.clip(v, -128, 127).astype(np.int8)


class Net:
    """Builds the op list, the weight blob and the NumPy forward pass together."""

    def __init__(self, name, H, W, C, seed):
        self.name, self.rng = name, np.random.default_rng(seed)
        self.tensors, self.ops, self.blob, self.vals, self.shapes, self.macs = [], [], bytearray(), [], [], []
        x = lcg_bytes(H * W * C).reshape(H * W, C)
        self.input = self.tensor(x, H, W)

    def tensor(self, v, H, W):
        self.tensors.append(v.shape[1])
        self.vals.append(v)
        self.shapes.append((H, W, v.shape[1]))
        return len(self.tensors) - 1

    def add_blob(self, arr):
        off = len(self.blob)
        data = np.ascontiguousarray(arr).tobytes()
        self.blob += data + b"\0" * (r16(len(data)) - len(data))
        return off

    def op(self, kind, in0, out, macs, K=0, stride=1, pad=0, shift=0, relu=0, in1=NONE16, W=0, w_off=NONE32,
           b_off=NONE32):
        self.ops.append((kind, relu, K, stride, pad, shift, 0, 0, 0, 0, in0, in1, out, W, w_off, b_off))
        self.macs.append(macs)

    def scale(self, acc_fn, fan_in):
        """Random weights for a layer of fan_in inputs per output, and the shift that puts its outputs'
        spread near a real network's (std ~ 30): -> (weights as int, shift)."""
        for wmax in (24, 12, 6, 3, 1):
            w = self.rng.integers(-wmax, wmax + 1, size=fan_in)
            acc = acc_fn(w)
            if np.abs(acc).max() < ACC_LIMIT * 0.9:
                break
        else:
            raise ValueError(f"{self.name}: a layer's sums overflow 20 bits even with weights of +-1")
        sd = acc.std() or 1.0
        return w, acc, max(0, int(np.round(np.log2(sd / 30))))

    def im2col(self, t, K, stride, pad):
        H, W, C = self.shapes[t]
        Ho, Wo = (H + 2 * pad - K) // stride + 1, (W + 2 * pad - K) // stride + 1
        Cy = r16(K * K * C)
        x = np.pad(self.vals[t].reshape(H, W, C), ((pad, pad), (pad, pad), (0, 0)))
        cols = [x[ky: ky + stride * (Ho - 1) + 1: stride, kx: kx + stride * (Wo - 1) + 1: stride] for ky in range(K)
                for kx in range(K)]
        y = np.zeros((Ho * Wo, Cy), np.int8)
        y[:, : K * K * C] = np.concatenate(cols, axis=2).reshape(Ho * Wo, K * K * C)
        out = self.tensor(y, Ho, Wo)
        self.op(IM2COL, t, out, 0, K=K, stride=stride, pad=pad, W=W)
        return out

    def conv1x1(self, t, N, relu=True, bias=True):
        H, W, C = self.shapes[t]
        x = self.vals[t].astype(np.int64)

        def acc_of(wflat):
            return x @ wflat.reshape(C, N)
        w, acc, _ = self.scale(acc_of, C * N)
        b = (self.rng.normal(0, acc.std() / 4, N).astype(np.int64) if bias else np.zeros(N, np.int64))
        b = np.clip(b, -ACC_LIMIT // 8, ACC_LIMIT // 8)
        acc = acc + b
        assert np.abs(acc).max() < ACC_LIMIT
        shift = max(0, int(np.round(np.log2((acc.std() or 1) / 30))))
        wf = w.reshape(C, N).astype(np.int8)
        w_off = self.add_blob(wf.reshape(C, N // 16, 16).transpose(1, 0, 2))
        b_off = self.add_blob(b.astype(np.int32)) if bias else NONE32
        out = self.tensor(out8(acc, shift, relu), H, W)
        self.op(CONV1X1, t, out, H * W * C * N, K=1, shift=shift, relu=int(relu), w_off=w_off, b_off=b_off)
        return out

    def dw(self, t, K=3, stride=1, pad=None, relu=True):
        H, W, C = self.shapes[t]
        pad = (K - 1) // 2 if pad is None else pad
        Ho, Wo = (H + 2 * pad - K) // stride + 1, (W + 2 * pad - K) // stride + 1
        x = np.pad(self.vals[t].reshape(H, W, C).astype(np.int64), ((pad, pad), (pad, pad), (0, 0)))

        def acc_of(wflat):
            wk = wflat.reshape(K * K, C)
            acc = np.zeros((Ho, Wo, C), np.int64)
            for ky in range(K):
                for kx in range(K):
                    acc += x[ky: ky + stride * (Ho - 1) + 1: stride, kx: kx + stride * (Wo - 1) + 1: stride] * wk[ky * K + kx]
            return acc.reshape(Ho * Wo, C)
        w, acc, shift = self.scale(acc_of, K * K * C)
        wf = w.reshape(K * K, C).astype(np.int8)
        w_off = self.add_blob(wf.reshape(K * K, C // 16, 16).transpose(1, 0, 2))
        out = self.tensor(out8(acc, shift, relu), Ho, Wo)
        self.op(DWCONV2D, t, out, Ho * Wo * C * K * K, K=K, stride=stride, pad=pad, shift=shift, relu=int(relu),
                W=W, w_off=w_off)
        return out

    def add(self, a, b):
        H, W, C = self.shapes[a]
        y = np.clip(self.vals[a].astype(np.int64) + self.vals[b], -128, 127).astype(np.int8)
        out = self.tensor(y, H, W)
        self.op(ADD, a, out, 0, in1=b)
        return out

    def mean(self, t):
        H, W, C = self.shapes[t]
        s = self.vals[t].astype(np.int64).sum(0)
        T = H * W
        y = np.clip(np.floor_divide(2 * s + T, 2 * T), -128, 127).astype(np.int8)[None]
        out = self.tensor(y, 1, 1)
        self.op(MEAN, t, out, 0)
        return out

    def bottleneck(self, t, expand, cout, stride):
        H, W, C = self.shapes[t]
        h = self.conv1x1(t, r16(int(round(C * expand)))) if expand != 1 else t
        h = self.dw(h, 3, stride)
        h = self.conv1x1(h, cout, relu=False)
        return self.add(h, t) if stride == 1 and C == cout else h

    def image(self):
        tensors = b"".join(TENSOR.pack(c, 0) for c in self.tensors)
        ops = b"".join(OP.pack(*o) for o in self.ops)
        t_off = r16(HDR.size)
        o_off = r16(t_off + len(tensors))
        b_off = r16(o_off + len(ops))
        out = len(self.tensors) - 1
        hdr = HDR.pack(MAGIC, VERSION, len(self.tensors), len(self.ops), self.input, out, t_off, o_off, b_off,
                       len(self.blob), self.tensors[out], 0, 0, 0, 0, 0)
        img = bytearray(b_off + len(self.blob))
        img[: len(hdr)] = hdr
        img[t_off: t_off + len(tensors)] = tensors
        img[o_off: o_off + len(ops)] = ops
        img[b_off:] = self.blob
        return bytes(img)


def mobilenetv2(res=224, seed=1):
    """MobileNetV2 1.0 (Sandler et al. 2018). Channel counts that aren't multiples of 16 (24 and the
    1000-way classifier) are padded up, as an export for MMRT would: ~4% more multiply-adds."""
    n = Net(f"mobilenetv2_{res}", res, res, 3, seed)
    t = n.conv1x1(n.im2col(n.input, 3, 2, 1), 32)   # the stem: 3x3 stride 2, as im2col + 1x1
    real = 32  # channels before padding: 24 padded to 32 still expands to 6 x 24 = 144
    for e, c, reps, s in [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1),
                          (6, 160, 3, 2), (6, 320, 1, 1)]:
        for i in range(reps):
            t = n.bottleneck(t, real * e / n.shapes[t][2], r16(c), s if i == 0 else 1)
            real = c
    t = n.conv1x1(t, 1280)
    t = n.mean(t)
    t = n.conv1x1(t, r16(1000), relu=False)
    return n


def mobilefacenet(res=112, seed=2):
    """MobileFaceNet (Chen et al. 2018): a face to a 128-d identity vector. PReLU as ReLU (the same cost
    as a lookup table on the board)."""
    n = Net(f"mobilefacenet_{res}", res, res, 3, seed)
    t = n.conv1x1(n.im2col(n.input, 3, 2, 1), 64)
    t = n.dw(t, 3, 1)
    for e, c, reps, s in [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]:
        for i in range(reps):
            t = n.bottleneck(t, e, c, s if i == 0 else 1)
    t = n.conv1x1(t, 512)
    t = n.dw(t, n.shapes[t][0], 1, pad=0, relu=False)  # GDConv: 7x7 depthwise over the whole map
    t = n.conv1x1(t, 128, relu=False, bias=False)
    return n


MODELS = {"mobilenetv2_224": mobilenetv2, "mobilefacenet_112": mobilefacenet}


# ---------------------------------------------------------------- the PC check
TRACE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.c_int)
ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
FREE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def build(flags, tag):
    lib = OUT / f"libmmrt_{tag}.so"
    src = ROOT / "mmrt"
    subprocess.check_call(["gcc", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC", *flags, "-o", str(lib),
                           str(src / "mmrt.c"), str(src / "mmrt_ref.c"), str(src / "mmrt_port.c")])
    return ctypes.CDLL(str(lib))


def check(net, img, lib, use_ref):
    libc = ctypes.CDLL(None)
    libc.aligned_alloc.restype = ctypes.c_void_p
    libc.aligned_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    alloc = ALLOC(lambda nb: libc.aligned_alloc(16, (nb + 15) // 16 * 16 or 16))
    free = FREE(lambda p: libc.free(p))
    bad = []

    def on_op(i, op, y, T, C):
        o = net.ops[i]
        got = np.ctypeslib.as_array(y, shape=(T * C,)).copy()
        want = net.vals[o[12]].reshape(-1)
        if got.shape != want.shape or not np.array_equal(got, want):
            bad.append((i, o[0], int((got != want).sum()) if got.shape == want.shape else "shape"))
    trace = TRACE(on_op)
    ctypes.c_void_p.in_dll(lib, "mmrt_trace").value = ctypes.cast(trace, ctypes.c_void_p).value
    ctypes.c_int.in_dll(lib, "mmrt_use_ref").value = use_ref
    buf = ctypes.create_string_buffer(img, len(img) + 16)
    model = ctypes.create_string_buffer(256)
    assert lib.mmrt_open(model, buf, len(img)) == 0
    x = np.ascontiguousarray(net.vals[net.input].reshape(-1))
    lib.mmrt_run.restype = ctypes.POINTER(ctypes.c_int8)
    lib.mmrt_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                             ALLOC, FREE]
    T_out = ctypes.c_int()
    H, W, _ = net.shapes[net.input]
    y = lib.mmrt_run(model, x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), H * W, ctypes.byref(T_out), alloc, free)
    assert y, "mmrt_run failed"
    got = np.ctypeslib.as_array(y, shape=(T_out.value * net.tensors[-1],)).copy()
    lib.mmrt_close(model, free)
    ctypes.c_void_p.in_dll(lib, "mmrt_trace").value = None
    return bad, np.array_equal(got, net.vals[-1].reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    libs = {}
    if a.check:
        libs = {"portable": build([], "port"), "p4-arith": build(["-DMMRT_C1", "-DMMRT_ROWDOT"], "c1")}
    for name in a.models:
        net = MODELS[name]()
        img = net.image()
        (OUT / f"{name}.mmrt").write_bytes(img)
        kinds = {DWCONV2D: "dwconv2d", CONV1X1: "conv1x1", IM2COL: "im2col", ADD: "add", MEAN: "mean"}
        info = {
            "model": name, "bytes": len(img), "ops": len(net.ops), "input": list(net.shapes[net.input]),
            "output": list(net.shapes[-1]), "mmac": round(sum(net.macs) / 1e6, 1),
            "mmac_by_kind": {k: round(sum(m for o, m in zip(net.ops, net.macs) if o[0] == kk) / 1e6, 1)
                             for kk, k in kinds.items()},
            "op_kinds": [kinds[o[0]] for o in net.ops],
            "op_mmac": [round(m / 1e6, 3) for m in net.macs],
            "act_peak_kb": round(max(net.vals[o[10]].size + net.vals[o[12]].size for o in net.ops) / 1024),
            "out_fnv": f"{fnv1a(net.vals[-1].tobytes()):08x}",
            "zeros_out": float((net.vals[-1] == 0).mean()),
        }
        (OUT / f"{name}.json").write_text(json.dumps(info, indent=1))
        print(f"{name}: {len(img) / 1e6:.2f} MB, {len(net.ops)} ops, {info['mmac']} M multiply-adds "
              f"{info['mmac_by_kind']}, peak activations {info['act_peak_kb']} KB, output fnv {info['out_fnv']}")
        for tag, lib in libs.items():
            for use_ref in (1, 0):
                bad, ok = check(net, img, lib, use_ref)
                what = "reference" if use_ref else "kernels"
                print(f"  {tag} {what}: {'bit-exact' if ok and not bad else 'MISMATCH'}"
                      + (f" (first ops: {bad[:5]})" if bad else ""))


if __name__ == "__main__":
    main()
